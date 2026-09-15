"""Minimal Redis-backed session state for cross-run persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


# Backward-compatible Phase 1 export.  This is no longer enforced as a hard
# limit: completed turns must reach ContextCompressor before any prefix is removed.
MAX_RECENT_MESSAGES = 20


def normalize_chat_history(messages: list | None) -> list[dict[str, str]]:
    """Accept conversation data only; do not import client system/tool instructions."""
    if messages is None:
        return []
    if not isinstance(messages, list) or len(messages) > 40:
        raise ValueError("chat_history must contain at most 40 messages")
    result = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("chat_history messages must be objects")
        role, content = message.get("role"), message.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str) or not content.strip():
            raise ValueError("chat_history requires non-empty user/assistant text messages")
        result.append({"role": role, "content": content.strip()})
    if sum(len(item["content"]) for item in result) > 20000:
        raise ValueError("chat_history exceeds 20000 characters")
    return result


@dataclass(slots=True)
class SessionState:
    session_id: str
    current_entity: str | None = None
    current_doc_scope: list[str] = field(default_factory=list)
    recent_messages: list[dict[str, str]] = field(default_factory=list)
    conversation_summary: str = ""
    last_run_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id.strip():
            raise ValueError("session_id must be a non-empty string")
        self.recent_messages = list(self.recent_messages)

    def append_message(self, role: str, content: str) -> None:
        self.recent_messages.append({"role": role, "content": content})

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, payload: str | bytes) -> "SessionState":
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        data = json.loads(payload)
        return cls(
            session_id=data["session_id"],
            current_entity=data.get("current_entity"),
            current_doc_scope=list(data.get("current_doc_scope") or []),
            recent_messages=list(data.get("recent_messages") or []),
            conversation_summary=data.get("conversation_summary") or "",
            last_run_id=data.get("last_run_id"),
        )


class RedisSessionStore:
    """Store one SessionState JSON document under one Redis key."""

    def __init__(
        self,
        redis_url: str,
        ttl_seconds: int,
        *,
        client: Any | None = None,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if client is None:
            import redis

            client = redis.Redis.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=1,
                socket_timeout=1,
            )
        self.client = client
        self.ttl_seconds = ttl_seconds

    @staticmethod
    def key(session_id: str) -> str:
        return f"finagent:session:{session_id}"

    def get(self, session_id: str) -> SessionState | None:
        payload = self.client.get(self.key(session_id))
        return None if payload is None else SessionState.from_json(payload)

    def set(self, state: SessionState) -> None:
        self.client.set(
            self.key(state.session_id),
            state.to_json(),
            ex=self.ttl_seconds,
        )

    def delete(self, session_id: str) -> None:
        self.client.delete(self.key(session_id))


__all__ = ["MAX_RECENT_MESSAGES", "RedisSessionStore", "SessionState"]
