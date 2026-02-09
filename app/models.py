from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class Citation(BaseModel):
    source: str = "NG12 PDF"
    page: int
    chunk_id: str
    excerpt: str


class AssessRequest(BaseModel):
    patient_id: str


class AssessResponse(BaseModel):
    patient_id: str
    category: Literal[
        "urgent_referral",
        "urgent_investigation",
        "no_urgent_action",
        "insufficient_evidence",
    ]
    rationale: str
    recommended_action: str
    citations: list[Citation] = Field(default_factory=list)


class ChatRequest(BaseModel):
    patient_id: str
    message: str


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatResponse(BaseModel):
    patient_id: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    history: list[Message] = Field(default_factory=list)


class HistoryResponse(BaseModel):
    patient_id: str
    history: list[Message] = Field(default_factory=list)
