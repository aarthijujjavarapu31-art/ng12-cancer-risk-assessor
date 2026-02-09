# NG12 Cancer Risk Assessor -- Reasoning Agent

This project implements a grounded clinical reasoning agent based on
NICE NG12 guidelines.

It combines structured patient data with unstructured clinical guidance
using a shared Retrieval-Augmented Generation (RAG) pipeline across:

-   Clinical decision support workflow (`/assess`)
-   Conversational chat interface (`/chat`)

All outputs are strictly grounded in retrieved NG12 excerpts.

------------------------------------------------------------------------

# System Overview

The system uses a shared RAG architecture:

-   Structured patient lookup
-   FAISS vector retrieval over NG12 chunks
-   Citation ranking
-   LLM reasoning grounded in retrieved evidence
-   Shared retriever reused across endpoints
-   Lightweight in-memory conversation storage

Both `/assess` and `/chat` use the same retriever and citation ranking
logic.

------------------------------------------------------------------------

# API Endpoints

## Health Check

GET `/health`

Returns:

``` json
{
  "status": "ok"
}
```

------------------------------------------------------------------------

## Clinical Assessment

POST `/assess`

Request:

``` json
{
  "patient_id": "PT-101"
}
```

Response includes:

-   category\
-   rationale\
-   recommended_action\
-   supporting NG12 citations

------------------------------------------------------------------------

## Conversational Interface

POST `/chat`

Request:

``` json
{
  "patient_id": "PT-101",
  "message": "Do I need urgent referral?"
}
```

Response includes:

-   grounded answer\
-   supporting citations\
-   full conversation history

------------------------------------------------------------------------

## View Conversation History

GET `/history/{patient_id}`

------------------------------------------------------------------------

## Clear Conversation History

DELETE `/history/{patient_id}`

------------------------------------------------------------------------

# Architecture

    app/
      agents/
        risk_assessor.py
        chat_agent.py
      rag/
        retriever.py
        vector_store.py
      tools/
        patient_lookup.py
      memory/
        session_store.py
      main.py

    data/
      ng12.index
      ng12_meta.json

------------------------------------------------------------------------

# Design Decisions

## Shared RAG Pipeline

Both endpoints use:

-   Same retriever\
-   Same citation ranking logic\
-   Same grounding rules

## Grounded Generation

The LLM:

-   Uses only retrieved NG12 excerpts\
-   Returns JSON only\
-   Avoids hallucinated thresholds or cancers\
-   Returns fallback message if unsupported

Fallback:

"I can't find that in the provided NG12 excerpts."

------------------------------------------------------------------------

# Installation

## Clone

    git clone <your-repo-url>
    cd ng12-assessor

## Create Virtual Environment

Windows:

    python -m venv .venv
    .venv\Scripts\activate

Mac/Linux:

    python3 -m venv .venv
    source .venv/bin/activate

## Install Dependencies

    pip install -r requirements.txt

## Configure Environment Variables

Create `.env` from `.env.example`:

    GOOGLE_CLOUD_PROJECT=your-project-id
    GOOGLE_CLOUD_LOCATION=us-central1
    NG12_MODEL=gemini-2.0-flash-001
    NG12_TOP_K=10
    NG12_CHAT_TOP_CITATIONS=3

Do NOT commit real credentials.

------------------------------------------------------------------------

## Authenticate Google Cloud

    gcloud auth application-default login
    gcloud config set project your-project-id
    gcloud auth application-default set-quota-project your-project-id

Ensure Vertex AI API enabled:

    gcloud services list --enabled | findstr aiplatform

------------------------------------------------------------------------

# Running the Server

    uvicorn app.main:app --reload

Server:

http://127.0.0.1:8000

Swagger:

http://127.0.0.1:8000/docs


# Future Improvements

-   Persist memory using Redis\
-   Add automated evaluation tests\
-   Add Docker support\
-   Improve retriever scoring
