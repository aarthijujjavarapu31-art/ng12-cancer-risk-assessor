from __future__ import annotations

import json
import os
import re
from typing import Any

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

from app.models import AssessResponse, Citation


PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
MODEL_NAME = os.getenv("NG12_MODEL", "gemini-2.0-flash-001")
TOP_K = int(os.getenv("NG12_TOP_K", "10"))


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _stable_query(patient: dict[str, Any]) -> str:
    age = patient.get("age", "")
    sex = patient.get("sex", "")
    duration = patient.get("duration", "")
    symptoms = patient.get("symptoms") or []
    symptoms_sorted = ", ".join(sorted([_norm(x) for x in symptoms]))

    return (
        "NICE NG12 suspected cancer recognition and referral.\n"
        f"Patient: age={age}, sex={sex}, duration={duration}\n"
        f"Symptoms: {symptoms_sorted}\n"
        "Question: Based on NG12, what is the recommended next action and urgency category?"
    )


def _score_citation(c: Citation, patient: dict[str, Any]) -> int:
    text = _norm(c.excerpt)

    symptoms = [_norm(s) for s in (patient.get("symptoms") or [])]
    has_weight_loss = any("weight loss" in s for s in symptoms)
    has_bowel_change = any("change in bowel" in s or "bowel habit" in s for s in symptoms)
    has_abdominal_pain = any("abdominal pain" in s for s in symptoms) or any("upper abdominal pain" in s for s in symptoms)

    score = 0

    # Prefer strong urgent language
    if "suspected cancer pathway" in text or "urgent investigation" in text or "urgent referral" in text:
        score += 60

    # Rule beats model: weight loss + bowel change => prefer pathway referral/investigation chunks
    if has_weight_loss and has_bowel_change:
        if "suspected cancer pathway" in text or "urgent investigation" in text or "urgent referral" in text:
            score += 80
        if "ct scan" in text or "direct access ct" in text:
            score -= 20

    # CT is relevant sometimes, but usually secondary for that combo
    if has_weight_loss and has_abdominal_pain and ("ct scan" in text or "direct access ct" in text):
        score += 15

    # Keyword overlap small boost
    for kw in ["weight loss", "change in bowel habit", "bowel habit", "abdominal pain", "upper abdominal pain", "ct scan"]:
        if kw in text:
            score += 3

    # Prefer slightly longer excerpts (but cap)
    score += min(len(text) // 200, 5)

    return score


def _rank_citations(citations: list[Citation], patient: dict[str, Any]) -> list[Citation]:
    return sorted(citations, key=lambda c: _score_citation(c, patient), reverse=True)


def _init_vertex() -> None:
    if not PROJECT:
        raise RuntimeError(
            "GOOGLE_CLOUD_PROJECT is not set. Example:\n"
            '  $env:GOOGLE_CLOUD_PROJECT="your-project-id"\n'
            '  $env:GOOGLE_CLOUD_LOCATION="us-central1"'
        )
    vertexai.init(project=PROJECT, location=LOCATION)


def _extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    return {}


def _map_category(cat: str) -> str:
    c = _norm(cat)
    if c in {"urgent_referral", "urgent referral"}:
        return "urgent_referral"
    if c in {"urgent_investigation", "urgent investigation"}:
        return "urgent_investigation"
    if c in {"no_urgent_action", "no urgent action", "routine"}:
        return "no_urgent_action"
    if c in {"insufficient_evidence", "insufficient evidence", "uncertain"}:
        return "insufficient_evidence"
    return "urgent_investigation"


def assess_patient(patient: dict[str, Any]) -> AssessResponse:
    from app.rag.retriever import NG12Retriever

    query = _stable_query(patient)

    try:
        retriever = NG12Retriever()
        citations = retriever.retrieve(query, top_k=TOP_K)
    except Exception as e:
        # Prevent flaky 500s from retrieval failures
        return AssessResponse(
            patient_id=str(patient.get("patient_id", "")),
            category="insufficient_evidence",
            rationale=f"Retrieval failed: {type(e).__name__}: {str(e)}",
            recommended_action="Retry retrieval or review NG12 guidance manually.",
            citations=[],
        )

    if not citations:
        return AssessResponse(
            patient_id=str(patient.get("patient_id", "")),
            category="insufficient_evidence",
            rationale="No relevant NG12 guidance could be retrieved for the given symptoms.",
            recommended_action="Review NG12 guidance manually and consider clinical review.",
            citations=[],
        )

    ranked = _rank_citations(citations, patient)
    best = ranked[0]

    # Deterministic generation config
    gen_cfg = GenerationConfig(
        temperature=0.0,
        top_p=1.0,
        candidate_count=1,
        max_output_tokens=512,
    )

    prompt = f"""
You are a clinical decision support assistant.
You MUST ground your answer only in the NG12 excerpt below.

Patient JSON:
{json.dumps(patient, indent=2)}

NG12 excerpt (primary evidence):
Source: {best.source}
Page: {best.page}
Chunk: {best.chunk_id}
Text:
{best.excerpt}

Return ONLY a JSON object with these keys:
- category: one of ["urgent_referral","urgent_investigation","no_urgent_action","insufficient_evidence"]
- rationale: short, quote/point to the excerpt wording
- recommended_action: short, actionable, aligned with the excerpt
""".strip()

    try:
        _init_vertex()
        model = GenerativeModel(MODEL_NAME)
        resp = model.generate_content(prompt, generation_config=gen_cfg)
        obj = _extract_json(getattr(resp, "text", "") or "")
    except Exception as e:
        # If model fails, still return something grounded in excerpt
        excerpt_norm = _norm(best.excerpt)
        if "suspected cancer pathway" in excerpt_norm or "urgent referral" in excerpt_norm:
            return AssessResponse(
                patient_id=str(patient.get("patient_id", "")),
                category="urgent_referral",
                rationale="NG12 excerpt indicates a suspected cancer pathway referral.",
                recommended_action="Refer using a suspected cancer pathway referral.",
                citations=[best],
            )
        if "urgent investigation" in excerpt_norm:
            return AssessResponse(
                patient_id=str(patient.get("patient_id", "")),
                category="urgent_investigation",
                rationale="NG12 excerpt indicates urgent investigation.",
                recommended_action="Offer urgent investigation.",
                citations=[best],
            )
        return AssessResponse(
            patient_id=str(patient.get("patient_id", "")),
            category="insufficient_evidence",
            rationale=f"Model failed: {type(e).__name__}: {str(e)}. Returning best available citation.",
            recommended_action="Follow NG12 guidance as per the cited excerpt.",
            citations=[best],
        )

    category = _map_category(str(obj.get("category", "")))
    rationale = str(obj.get("rationale") or "").strip()
    action = str(obj.get("recommended_action") or "").strip()

    if not rationale:
        rationale = "Recommendation is grounded in the retrieved NG12 excerpt."
    if not action:
        excerpt_norm = _norm(best.excerpt)
        if "suspected cancer pathway" in excerpt_norm or "urgent referral" in excerpt_norm:
            action = "Refer using a suspected cancer pathway referral."
            category = "urgent_referral"
        elif "urgent investigation" in excerpt_norm:
            action = "Offer urgent investigation."
            category = "urgent_investigation"
        else:
            action = "Follow NG12 guidance as per the cited excerpt."

    return AssessResponse(
        patient_id=str(patient.get("patient_id", "")),
        category=category,
        rationale=rationale,
        recommended_action=action,
        citations=[best],
    )
