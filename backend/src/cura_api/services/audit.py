"""Append-only audit log helpers."""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from cura_api.db import utcnow
from cura_api.models import AuditEvent

SYSTEM_ROLE = "system"


def record(session: Session, actor_id: str, actor_role: str, action: str, entity_type: str,
           entity_id: str | None, details: dict[str, Any] | None = None) -> AuditEvent:
    event = AuditEvent(occurred_at=utcnow(), actor_id=actor_id, actor_role=actor_role, action=action,
                       entity_type=entity_type, entity_id=entity_id, details=details or {})
    session.add(event)
    return event
