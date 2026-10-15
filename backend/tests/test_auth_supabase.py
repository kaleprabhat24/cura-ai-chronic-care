"""Supabase-mode authentication: signature, issuer, audience, expiry and claims checks, mapping to
provisioned app users, fail-closed behaviour and server-side role enforcement.

Real Supabase is not contacted: tokens are signed with locally generated keys and the verifier is
given those public keys in place of the project's JWKS endpoint (the same code path PyJWT uses).
"""

from __future__ import annotations

import shutil
import time

import jwt
import pytest
from conftest import make_settings
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi.testclient import TestClient

from cura_api.auth import SupabaseTokenVerifier
from cura_api.db import make_engine, make_session_factory
from cura_api.main import create_app
from cura_api.services.users import UserError, add_user, list_users, update_user

PROJECT = "https://testproj.supabase.co"
ISSUER = f"{PROJECT}/auth/v1"
LEGACY = "legacy-test-secret-not-a-real-secret-0123456789"


class StubJWKS:
    """Stands in for jwt.PyJWKClient (which would fetch <project>/auth/v1/.well-known/jwks.json)."""

    def __init__(self, public_keys: dict[str, object]) -> None:
        self.keys = {kid: jwt.PyJWK.from_dict(jwt.algorithms.ECAlgorithm.to_jwk(k, as_dict=True)
                                              | {"kid": kid, "alg": "ES256", "use": "sig"})
                     for kid, k in public_keys.items()}

    def get_signing_key_from_jwt(self, token: str):
        kid = jwt.get_unverified_header(token).get("kid")
        if kid not in self.keys:
            raise jwt.PyJWKClientError(f"unable to find a signing key that matches {kid!r}")
        return self.keys[kid]


@pytest.fixture(scope="module")
def keys():
    return {"good": ec.generate_private_key(ec.SECP256R1()), "other": ec.generate_private_key(ec.SECP256R1())}


def token(keys, *, sub="11111111-2222-3333-4444-555555555555", email="coord@example.org",
          role="authenticated", iss=ISSUER, aud="authenticated", exp_in=3600, kid="kid-1",
          signer="good", alg="ES256", secret=None, extra=None) -> str:
    now = int(time.time())
    claims = {"sub": sub, "email": email, "role": role, "iss": iss, "aud": aud, "iat": now - 5,
              "exp": now + exp_in, **(extra or {})}
    if alg == "HS256":
        return jwt.encode(claims, secret, algorithm="HS256")
    return jwt.encode(claims, keys[signer], algorithm=alg, headers={"kid": kid})


@pytest.fixture(scope="module")
def sb(env, keys, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("supabase")
    db = tmp / "auth.db"
    shutil.copy(env.root / "cura_test.db", db)
    url = f"sqlite:///{db.as_posix()}"
    engine = make_engine(url)
    factory = make_session_factory(engine)
    add_user(factory, "coord@example.org", "coordinator", "Casey Coordinator")
    add_user(factory, "Analyst@Example.org", "analyst")
    add_user(factory, "admin@example.org", "admin")
    add_user(factory, "former@example.org", "coordinator")
    update_user(factory, "former@example.org", active=False)
    settings = make_settings(env.root, url, env.data_dir, env.bundle_dir, auth_mode="supabase",
                             supabase_url=PROJECT)
    verifier = SupabaseTokenVerifier(ISSUER, "authenticated", None, LEGACY,
                                     jwks_client=StubJWKS({"kid-1": keys["good"].public_key()}))
    with TestClient(create_app(settings, token_verifier=verifier)) as c:
        yield c, factory
    engine.dispose()


def get(client, path, tok=None, headers=None):
    h = dict(headers or {})
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return client.get(f"/api/v1{path}", headers=h)


def code(r) -> tuple[int, str]:
    return r.status_code, r.json()["error"]["code"]


def test_valid_token_maps_to_provisioned_role(sb, keys):
    c, _ = sb
    r = get(c, "/me", token(keys))
    assert r.status_code == 200
    assert r.json() == {"user_id": "coord@example.org", "role": "coordinator", "auth_mode": "supabase",
                        "display_name": "Casey Coordinator"}
    assert get(c, "/worklist?page_size=1", token(keys)).status_code == 200


def test_identity_is_bound_on_first_sign_in(sb, keys):
    c, factory = sb
    sub = "aaaaaaaa-0000-0000-0000-000000000001"
    assert get(c, "/me", token(keys, sub=sub, email="analyst@example.org")).json()["role"] == "analyst"
    # later tokens for the same identity work even if the email claim changes
    assert get(c, "/me", token(keys, sub=sub, email="renamed@example.org")).json()["role"] == "analyst"
    assert any(u["email"] == "analyst@example.org" and u["linked"] for u in list_users(factory))
    # a different identity cannot claim an email that is already bound
    assert code(get(c, "/me", token(keys, sub="bbbbbbbb-0000-0000-0000-000000000002",
                                    email="analyst@example.org"))) == (403, "not_provisioned")


@pytest.mark.parametrize("kwargs,expected", [
    ({"exp_in": -120}, "invalid_token"),                                   # expired (beyond leeway)
    ({"iss": "https://evil.supabase.co/auth/v1"}, "invalid_token"),        # wrong issuer
    ({"aud": "anon"}, "invalid_token"),                                    # wrong audience
    ({"signer": "other"}, "invalid_token"),                                # signature not by the project
    ({"kid": "unknown-kid"}, "invalid_token"),                             # key not in JWKS
    ({"role": "anon"}, "invalid_token"),                                   # not a user session
    ({"role": "service_role"}, "invalid_token"),
    ({"alg": "HS256", "secret": "wrong-secret-wrong-secret-wrong-secret!!"}, "invalid_token"),
])
def test_invalid_tokens_rejected(sb, keys, kwargs, expected):
    c, _ = sb
    assert code(get(c, "/me", token(keys, **kwargs))) == (401, expected)


def test_missing_malformed_and_unsigned_tokens_rejected(sb, keys):
    c, _ = sb
    assert code(get(c, "/me")) == (401, "unauthenticated")
    assert code(get(c, "/me", "not-a-jwt")) == (401, "invalid_token")
    unsigned = jwt.encode({"sub": "x", "role": "authenticated", "iss": ISSUER, "aud": "authenticated",
                           "exp": int(time.time()) + 60, "iat": int(time.time())}, None, algorithm="none")
    assert code(get(c, "/me", unsigned)) == (401, "invalid_token")
    # development identity headers are ignored in Supabase mode
    assert code(get(c, "/me", headers={"X-Dev-User": "x", "X-Dev-Role": "admin"})) == (401, "unauthenticated")


def test_legacy_hs256_secret_supported_when_configured(sb, keys):
    c, _ = sb
    tok = token(keys, alg="HS256", secret=LEGACY, sub="cccccccc-0000-0000-0000-000000000003",
                email="admin@example.org")
    assert get(c, "/me", tok).json()["role"] == "admin"
    no_secret = SupabaseTokenVerifier(ISSUER, "authenticated", None, None, jwks_client=StubJWKS({}))
    with pytest.raises(Exception, match="unsupported token algorithm"):
        no_secret.verify(tok)


def test_unprovisioned_and_deactivated_users_fail_closed(sb, keys):
    c, _ = sb
    assert code(get(c, "/me", token(keys, sub="dddddddd-0000-0000-0000-000000000004",
                                    email="stranger@example.org"))) == (403, "not_provisioned")
    assert code(get(c, "/me", token(keys, sub="eeeeeeee-0000-0000-0000-000000000005",
                                    email="former@example.org"))) == (403, "forbidden")


def test_roles_enforced_server_side_with_tokens(sb, keys):
    c, _ = sb
    analyst = token(keys, sub="aaaaaaaa-0000-0000-0000-000000000001", email="analyst@example.org")
    assert code(get(c, "/worklist", analyst)) == (403, "forbidden")
    assert code(get(c, "/patients/pt_000001/fhir", analyst)) == (403, "forbidden")
    assert get(c, "/dashboard/summary", analyst).status_code == 200
    coord = token(keys)
    assert code(get(c, "/audit-events", coord)) == (403, "forbidden")
    admin = token(keys, sub="cccccccc-0000-0000-0000-000000000003", email="admin@example.org")
    users = get(c, "/admin/users", admin).json()
    assert {u["email"] for u in users} >= {"coord@example.org", "admin@example.org"}
    r = c.post("/api/v1/episodes/ep_x/outreach-attempts", json={"channel": "phone", "outcome": "no_answer"},
               headers={"Authorization": f"Bearer {analyst}"})
    assert code(r) == (403, "forbidden")


def test_user_provisioning_rules(sb):
    _, factory = sb
    with pytest.raises(UserError):
        add_user(factory, "coord@example.org", "coordinator")
    with pytest.raises(UserError):
        add_user(factory, "not-an-email", "coordinator")
    with pytest.raises(UserError):
        add_user(factory, "x@example.org", "superuser")
    assert update_user(factory, "admin@example.org", role="analyst")["role"] == "analyst"
    update_user(factory, "admin@example.org", role="admin")
