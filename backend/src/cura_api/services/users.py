"""Provisioning of application users and roles (used by `cura-api users ...`)."""

from __future__ import annotations

import re

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from cura_api import models as m
from cura_api.auth import ROLES
from cura_api.db import utcnow
from cura_api.services import audit

EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class UserError(ValueError):
    pass


def _find(session: Session, email: str) -> m.AppUser | None:
    query = select(m.AppUser).where(func.lower(m.AppUser.email) == email.lower())
    return session.execute(query).scalar_one_or_none()


def add_user(factory: sessionmaker[Session], email: str, role: str, display_name: str | None = None,
             actor: str = "system:cli") -> dict:
    email = email.strip().lower()
    if not EMAIL.match(email):
        raise UserError(f"invalid email: {email!r}")
    if role not in ROLES:
        raise UserError(f"role must be one of {ROLES}")
    with factory() as s, s.begin():
        if _find(s, email):
            raise UserError(f"user {email} already exists (use set-role)")
        now = utcnow()
        u = m.AppUser(email=email, display_name=display_name, role=role, is_active=True,
                      created_at=now, updated_at=now)
        s.add(u)
        s.flush()
        audit.record(s, actor, audit.SYSTEM_ROLE, "user.add", "app_user", str(u.id),
                     {"email": email, "role": role})
        return {"id": u.id, "email": email, "role": role}


def update_user(factory: sessionmaker[Session], email: str, *, role: str | None = None,
                active: bool | None = None, actor: str = "system:cli") -> dict:
    if role is not None and role not in ROLES:
        raise UserError(f"role must be one of {ROLES}")
    with factory() as s, s.begin():
        u = _find(s, email)
        if u is None:
            raise UserError(f"no user {email}")
        changes = {}
        if role is not None and role != u.role:
            changes["role"] = [u.role, role]
            u.role = role
        if active is not None and active != u.is_active:
            changes["is_active"] = [u.is_active, active]
            u.is_active = active
        u.updated_at = utcnow()
        audit.record(s, actor, audit.SYSTEM_ROLE, "user.update", "app_user", str(u.id),
                     {"email": u.email, "changes": changes})
        return {"id": u.id, "email": u.email, "role": u.role, "is_active": u.is_active}


def list_users(factory: sessionmaker[Session]) -> list[dict]:
    with factory() as s:
        return [{"email": u.email, "role": u.role, "is_active": u.is_active,
                 "linked": u.auth_user_id is not None}
                for u in s.execute(select(m.AppUser).order_by(m.AppUser.email)).scalars()]
