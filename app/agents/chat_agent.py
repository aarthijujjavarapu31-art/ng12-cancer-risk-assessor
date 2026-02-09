from __future__ import annotations

import json
import os
import re
from typing import Any

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

from app.models import Citation


MODEL_NAME_DEFAULT = "gemini-2.0-flash-001"


# ----------------------------
# Utilities
# ----------------------------

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _init_vertex() -> None:
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

    if not project:
        raise RuntimeError(
            "GOOGLE_CLOUD_PROJECT is not set.\n"
            'Example:\n'
            '  $env:GOOGLE_CLOUD_PROJECT="your-project-id"\n'
            '  $env:GOOGLE_CLOUD_LOCATION="us-central1"'
        )

    vertexai.init(project=project, location=location)


def _extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()

    # Direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # Extract first JSON object from text
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    return {}


# ----------------------------
# Citation Ranking
# ----------------------------

def _score_citation(c: Citation, patient: dict[str, Any], message: str) -> int:
    text = _norm(c.excerpt)
    msg = _norm(message)

    symptoms = [_norm(s) for s in (patient.get("symptoms") or [])]
    findings = [_norm(f) for f in (patient.get("findings") or [])]

    patient_text = " ".join(symptoms + findings)

    score = 0

    # Strong referral signals
    if "refer using a suspected cancer pathway referral" in text:
        score += 120
    if "suspected cancer pathway referral" in text:
        score += 90
    if "offer urgent investigation" in text:
        score += 80
    if "urgent referral" in text:
        score += 60

    # Symptom match boost
    for kw in [
        "weight loss",
        "change in bowel",
        "bowel habit",
        "abdominal pain",
        "upper abdominal pain",
    ]:
        if kw in text and (kw in patient_text or kw in msg):
            score += 10

    # Penalize generic intro sections
    if "recommendations organised by symptom" in text:
        score -= 40

    # Slight specificity boost
    score += min(len(text) // 250, 6)

    return score


def _rank_citations(
    citations: list[Citation],
    patient: dict[str, Any],
    message: str,
) -> list[Citation]:
    return sorted(
        citations,
        key=lambda c: _score_citation(c, patient, message),
        reverse=True,
    )


# ----------------------------
# Patient-Guideline Gating
# ----------------------------

def _citation_supported_by_patient(c: Citation, patient: dict[str, Any]) -> bool:
    """
    Prevent LLM from citing criteria not present in patient.
    Example: splenomegaly, fever, night sweats, etc.
    """
    text = _norm(c.excerpt)

    patient_terms = " ".join(
        _norm(x)
        for x in (
            (patient.get("symptoms") or [])
            + (patient.get("findings") or [])
            + (patient.get("investigations") or [])
        )
    )

    gated_terms = [
        "splenomegaly",
        "lymphadenopathy",
        "night sweats",
        "fever",
        "pruritus",
    ]

    for term in gated_terms:
        if term in text and term not in patient_terms:
            return False

    return True


# ----------------------------
# Main Chat Function
# ----------------------------

def answer_question(
    patient: dict[str, Any],
    message: str,
) -> tuple[str, list[Citation]]:
    """
    Returns:
        (answer, citations)

    Notes:
    - No session_store writes here (main.py handles memory)
    - Fully grounded in retrieved excerpts only
    """

    from app.rag.retriever import NG12Retriever  # local import

    top_k = int(os.getenv("NG12_TOP_K", "10"))
    top_citations = int(os.getenv("NG12_CHAT_TOP_CITATIONS", "3"))
    model_name = os.getenv("NG12_MODEL", MODEL_NAME_DEFAULT)

    # ----------------------------
    # Build Retrieval Query
    # ----------------------------

    age = patient.get("age")
    sex = patient.get("sex") or patient.get("gender")
    symptoms = patient.get("symptoms") or []
    findings = patient.get("findings") or []

    symptom_text = "; ".join(str(s) for s in symptoms if s)
    finding_text = "; ".join(str(f) for f in findings if f)

    query = (
        "NG12 suspected cancer pathway referral urgent investigation. "
        f"Patient age {age}, sex {sex}. "
        f"Symptoms: {symptom_text}. Findings: {finding_text}. "
        f"Question: {message}"
    ).strip()

    retriever = NG12Retriever()
    raw = retriever.retrieve(query, top_k=top_k)

    # ----------------------------
    # Rank + Gate
    # ----------------------------

    ranked = _rank_citations(raw, patient, message)

    filtered = [
        c for c in ranked
        if _citation_supported_by_patient(c, patient)
    ]

    citations = filtered[: max(1, top_citations)]

    if not citations:
        return (
            "I couldn’t retrieve a relevant NG12 excerpt for that question.",
            [],
        )

    # ----------------------------
    # Build Prompt
    # ----------------------------

    evidence_block = "\n\n".join(
        [
            f"Evidence {i+1}:\n"
            f"Source: {c.source}\n"
            f"Page: {c.page}\n"
            f"Chunk: {c.chunk_id}\n"
            f"Text:\n{c.excerpt}"
            for i, c in enumerate(citations)
        ]
    )

    _init_vertex()
    model = GenerativeModel(model_name)

    prompt = f"""
You are a clinical decision support assistant.

RULES:
- Use ONLY the provided Evidence.
- If not supported by Evidence, say exactly:
  "I can’t find that in the provided NG12 excerpts."
- Do NOT invent findings.
- Keep the answer concise and specific to the patient.

Patient:
{json.dumps(patient, indent=2)}

User question:
{message}

Evidence:
{evidence_block}

Return ONLY JSON:
{{
  "answer": "string"
}}
""".strip()

    gen_cfg = GenerationConfig(
        temperature=0.0,
        top_p=1.0,
        candidate_count=1,
        max_output_tokens=512,
    )

    resp = model.generate_content(prompt, generation_config=gen_cfg)
    obj = _extract_json(getattr(resp, "text", "") or "")

    answer = str(obj.get("answer") or "").strip()

    if not answer:
        answer = "I can’t find that in the provided NG12 excerpts."

    return answer, citations
