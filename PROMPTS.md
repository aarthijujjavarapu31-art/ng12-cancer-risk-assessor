# PROMPTS (Assessment Mode)

## Goal
Given a patient_id, retrieve structured patient data and retrieve relevant NG12 guideline chunks from the vector store, then produce a grounded referral assessment.

## System rules (high level)
- Use ONLY the retrieved NG12 passages to justify clinical thresholds/criteria.
- If evidence is insufficient, say so and do not invent thresholds.
- Always return JSON in the API schema.
- Always include citations for any clinical statement.

## Inputs provided to the model
- Patient demographics + symptoms (structured)
- Retrieved guideline chunks (top_k)
- Output schema (AssessResponse)

## Output format
- classification: Urgent Referral / Urgent Investigation / Not met
- rationale: short explanation grounded in evidence
- citations: list of {page, chunk_id, excerpt}

