from __future__ import annotations

from typing import Dict, List

from app.models import Message

_STORE: Dict[str, List[Message]] = {}


def add_message(patient_id: str, role: str, content: str) -> None:
    if patient_id not in _STORE:
        _STORE[patient_id] = []
    _STORE[patient_id].append(Message(role=role, content=content))


def get_history(patient_id: str) -> list[Message]:
    return list(_STORE.get(patient_id, []))


def clear(patient_id: str) -> None:
    _STORE.pop(patient_id, None)
