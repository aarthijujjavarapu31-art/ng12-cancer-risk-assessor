# app/main.py
from __future__ import annotations

import os
import traceback
from typing import Any

from fastapi import FastAPI, HTTPException

from app.models import AssessRequest, AssessResponse, ChatRequest, ChatResponse, HistoryResponse
from app.tools.patient_lookup import get_patient
from app.agents.risk_assessor import assess_patient
from app.agents.chat_agent import answer_question
from app.memory.session_store import add_message, get_history, clear as clear_history

app = FastAPI(title="NG12 Cancer Risk Assessor", version="1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.on_event("startup")
def startup_log() -> None:
    print("[INFO] Starting NG12 server")
    print("[INFO] GOOGLE_CLOUD_PROJECT =", os.getenv("GOOGLE_CLOUD_PROJECT"))
    print("[INFO] GOOGLE_CLOUD_LOCATION =", os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"))
    print("[INFO] NG12_MODEL =", os.getenv("NG12_MODEL", "gemini-2.0-flash-001"))
    print("[INFO] NG12_TOP_K =", os.getenv("NG12_TOP_K", "10"))


@app.post("/assess", response_model=AssessResponse)
def assess(req: AssessRequest) -> AssessResponse:
    patient = get_patient(req.patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    try:
        return assess_patient(patient)
    except Exception as e:
        print("[ERROR] /assess failed")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    patient = get_patient(req.patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    # store user message first
    add_message(req.patient_id, role="user", content=req.message)

    try:
        result = answer_question(patient=patient, message=req.message)

        answer: str | None = None
        citations: list[Any] = []

        # Common case: (answer, citations)
        if isinstance(result, tuple):
            if len(result) >= 1:
                answer = result[0]
            if len(result) >= 2:
                citations = result[1] or []

        # If someone returns a model/dict
        elif isinstance(result, dict):
            answer = result.get("answer") or result.get("response") or result.get("text")
            citations = result.get("citations") or []
        else:
            answer = getattr(result, "answer", None) or getattr(result, "text", None)
            citations = getattr(result, "citations", []) or []

        if not answer:
            raise ValueError("answer_question() did not return an answer in a supported format.")

    except Exception as e:
        print("[ERROR] /chat failed")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    # store assistant response
    add_message(req.patient_id, role="assistant", content=str(answer))

    return ChatResponse(
        patient_id=req.patient_id,
        answer=str(answer),
        citations=citations,
        history=get_history(req.patient_id),
    )


@app.get("/history/{patient_id}", response_model=HistoryResponse)
def history(patient_id: str) -> HistoryResponse:
    return HistoryResponse(patient_id=patient_id, history=get_history(patient_id))


@app.delete("/history/{patient_id}")
def delete_history(patient_id: str) -> dict[str, Any]:
    clear_history(patient_id)
    return {"status": "cleared", "patient_id": patient_id}
