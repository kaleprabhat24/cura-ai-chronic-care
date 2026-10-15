"""Authentication and role-based authorization.

Roles:
  coordinator — worklist, patient records, risk scores and explanations, recording outreach, FHIR export
  analyst     — aggregate dashboard, outcomes, model and data-quality information (no patient views)
  admin       — everything, including scoring runs, users and the audit log

Two modes (`AUTH_MODE`):
  * `supabase` — the caller sends `Authorization: Bearer <Supabase access token>`. The token's
    signature is verified against the project's published signing keys (JWKS: RS256/ES256), or
    against the legacy shared JWT secret (HS256) only if `SUPABASE_JWT_SECRET` is configured. Issuer,
    audience, expiry, `sub` and `role == "authenticated"` are checked. The verified identity must
    match an active row in `app_users`, which supplies the role. Anything else fails closed (401/403).
    Role headers are ignored.
  * `dev` — local development only. The caller states an identity in `X-Dev-User` / `X-Dev-Role`.
    It refuses to start with `ENVIRONMENT=production`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Annotated, Any, Literal, get_args

import jwt
from fastapi import Depends, Header, Request
from sqlalchemy import func, select

from cura_api import models as m
from cura_api.db import utcnow
from cura_api.errors import ApiError

Role = Literal["coordinator", "analyst", "admin"]
ROLES: tuple[str, ...] = get_args(Role)
USER_ID = re.compile(r"^[A-Za-z0-9._@:-]{1,80}$")
ASYMMETRIC_ALGS = ("RS256", "ES256")
CLOCK_SKEW_SECONDS = 30


@dataclass(frozen=True)
class Principal:
    user_id: str
    role: str
    auth_mode: str
    display_name: str | None = None


class TokenError(Exception):
    pass


class SupabaseTokenVerifier:
    """Verifies Supabase Auth access tokens. `jwks_client` may be injected (tests)."""

    def __init__(self, issuer: str, audience: str, jwks_url: str | None,
                 legacy_secret: str | None = None, jwks_client: Any | None = None) -> None:
        self.issuer, self.audience, self.legacy_secret = issuer, audience, legacy_secret
        self.jwks_client = jwks_client or (jwt.PyJWKClient(jwks_url, cache_keys=True, lifespan=600)
                                           if jwks_url else None)

    def verify(self, token: str) -> dict[str, Any]:
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError as exc:
            raise TokenError("malformed token") from exc
        alg = header.get("alg")
        if alg in ASYMMETRIC_ALGS:
            if self.jwks_client is None:
                raise TokenError("no signing keys configured")
            try:
                key = self.jwks_client.get_signing_key_from_jwt(token).key
            except jwt.PyJWTError as exc:
                raise TokenError("signing key not found") from exc
        elif alg == "HS256" and self.legacy_secret:
            key = self.legacy_secret
        else:
            raise TokenError(f"unsupported token algorithm {alg!r}")
        try:
            claims = jwt.decode(token, key, algorithms=[alg], audience=self.audience, issuer=self.issuer,
                                leeway=CLOCK_SKEW_SECONDS,
                                options={"require": ["exp", "iat", "sub", "iss", "aud"]})
        except jwt.ExpiredSignatureError as exc:
            raise TokenError("token expired") from exc
        except jwt.PyJWTError as exc:
            raise TokenError(f"invalid token: {exc}") from exc
        if claims.get("role") != "authenticated":
            raise TokenError("token is not an authenticated user session")
        return claims


def build_verifier(settings) -> SupabaseTokenVerifier:
    secret = settings.supabase_jwt_secret.get_secret_value() if settings.supabase_jwt_secret else None
    return SupabaseTokenVerifier(settings.supabase_issuer, settings.supabase_jwt_audience,
                                 settings.resolved_jwks_url, secret)


def _supabase_principal(request: Request, authorization: str | None) -> Principal:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise ApiError(401, "unauthenticated", "Sign in required (Authorization: Bearer <token>).")
    token = authorization.split(" ", 1)[1].strip()
    try:
        claims = request.app.state.token_verifier.verify(token)
    except TokenError as exc:
        raise ApiError(401, "invalid_token", "The access token is invalid or expired.",
                       {"reason": str(exc)}) from exc
    sub = str(claims["sub"])
    email = (claims.get("email") or "").strip().lower()
    with request.app.state.session_factory() as session, session.begin():
        user = session.execute(select(m.AppUser).where(m.AppUser.auth_user_id == sub)).scalar_one_or_none()
        if user is None and email:
            # first sign-in: bind the verified Supabase identity to the provisioned email
            user = session.execute(select(m.AppUser).where(func.lower(m.AppUser.email) == email,
                                                           m.AppUser.auth_user_id.is_(None))).scalar_one_or_none()
            if user is not None:
                user.auth_user_id = sub
        if user is None:
            raise ApiError(403, "not_provisioned", "This account has no Cura AI access. Ask an "
                           "administrator to add your email with a role.")
        if not user.is_active or user.role not in ROLES:
            raise ApiError(403, "forbidden", "This account is deactivated.")
        user.last_sign_in_at = utcnow()
        return Principal(user.email, user.role, "supabase", user.display_name)


def get_principal(
    request: Request,
    authorization: Annotated[str | None, Header(description="Supabase mode: Bearer access token")] = None,
    x_dev_user: Annotated[str | None, Header(description="Dev mode only: user id")] = None,
    x_dev_role: Annotated[str | None, Header(description="Dev mode only: role "
                                                         "(coordinator | analyst | admin)")] = None,
) -> Principal:
    settings = request.app.state.settings
    if settings.auth_mode == "supabase":
        return _supabase_principal(request, authorization)
    if settings.auth_mode != "dev" or settings.environment == "production":  # pragma: no cover
        raise ApiError(503, "auth_not_configured", "Authentication is not configured.")
    if not x_dev_user or not x_dev_role:
        raise ApiError(401, "unauthenticated",
                       "Development authentication: send X-Dev-User and X-Dev-Role headers.")
    if not USER_ID.match(x_dev_user) or x_dev_role not in ROLES:
        raise ApiError(401, "unauthenticated", "Invalid development user id or role.",
                       {"allowed_roles": list(ROLES)})
    return Principal(x_dev_user, x_dev_role, "dev")


def require(*roles: str):
    def dependency(principal: Annotated[Principal, Depends(get_principal)]) -> Principal:
        if principal.role not in roles:
            raise ApiError(403, "forbidden", f"Role '{principal.role}' may not perform this action.",
                           {"required_roles": list(roles)})
        return principal
    return dependency


ANY_ROLE = require("coordinator", "analyst", "admin")
CLINICAL = require("coordinator", "admin")      # patient-identifiable views, outreach writes, FHIR export
ANALYTICS = require("analyst", "admin")         # model and data-quality internals
ADMIN = require("admin")
