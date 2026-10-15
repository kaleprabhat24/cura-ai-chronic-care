"""Authentication boundary and role-based authorization.

Roles:
  coordinator — worklist, patient records, risk scores and explanations, recording outreach
  analyst     — aggregate dashboard, outcomes, model and data-quality information (no patient views)
  admin       — everything, including scoring runs and the audit log

Only a **development** authentication mode exists today (`AUTH_MODE=dev`). The caller states an
identity in the `X-Dev-User` and `X-Dev-Role` headers. This is not secure, is meant for local work
only, and the application refuses to start with it when `ENVIRONMENT=production`. Real token
validation (Supabase JWT) is intentionally not implemented in this milestone. `get_principal` is the
single place where it will plug in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Annotated, Literal, get_args

from fastapi import Depends, Header, Request

from cura_api.errors import ApiError

Role = Literal["coordinator", "analyst", "admin"]
ROLES: tuple[str, ...] = get_args(Role)
USER_ID = re.compile(r"^[A-Za-z0-9._@:-]{1,80}$")


@dataclass(frozen=True)
class Principal:
    user_id: str
    role: str
    auth_mode: str


def get_principal(
    request: Request,
    x_dev_user: Annotated[str | None, Header(description="Development auth only: user id")] = None,
    x_dev_role: Annotated[str | None, Header(description="Development auth only: role "
                                                         "(coordinator | analyst | admin)")] = None,
) -> Principal:
    settings = request.app.state.settings
    if settings.auth_mode != "dev":  # pragma: no cover - start-up refuses other modes
        raise ApiError(503, "auth_not_configured", "Production authentication is not implemented.")
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
CLINICAL = require("coordinator", "admin")      # patient-identifiable views and outreach writes
ANALYTICS = require("analyst", "admin")         # model and data-quality internals
ADMIN = require("admin")
