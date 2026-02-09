from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Absolute path to /data/patients.json from this file:
# app/tools/patient_lookup.py -> app/tools -> app -> project root
ROOT = Path(__file__).resolve().parents[2]
PATIENTS_PATH = ROOT / "data" / "patients.json"


def get_patient(patient_id: str) -> dict[str, Any] | None:
    if not PATIENTS_PATH.exists():
        return None

    raw = PATIENTS_PATH.read_text(encoding="utf-8").strip()
    if not raw:
        # empty file -> treat as no patients instead of crashing
        return None

    try:
        patients = json.loads(raw)
    except json.JSONDecodeError:
        return None

    if isinstance(patients, dict):
        # allow {"PT-101": {...}}
        p = patients.get(patient_id)
        return p if isinstance(p, dict) else None

    if isinstance(patients, list):
        # allow [{"patient_id": "...", ...}]
        for p in patients:
            if isinstance(p, dict) and p.get("patient_id") == patient_id:
                return p

    return None
