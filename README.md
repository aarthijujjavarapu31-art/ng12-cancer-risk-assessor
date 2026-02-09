# NG12 Cancer Risk Assessor --- Reasoning Agent

## Overview

This project implements a grounded clinical reasoning agent based on the
NICE NG12 cancer referral guidelines.

It combines structured patient data with unstructured clinical guidance
using a shared Retrieval-Augmented Generation (RAG) pipeline across:

-   Clinical decision support workflow (`/assess`)
-   Conversational chat interface (`/chat`)

All outputs are strictly grounded in retrieved NG12 excerpts and include
citations.

------------------------------------------------------------------------

## System Overview

The system uses a shared RAG architecture:

-   Structured patient lookup (simulated database)
-   PDF ingestion and embedding pipeline
-   FAISS vector store for retrieval
-   Citation ranking
-   LLM reasoning grounded only in retrieved evidence
-   Shared retriever reused across endpoints
-   Lightweight in-memory conversation storage (per patient)
-   Embeddings are generated during ingestion and stored in a local FAISS index for reuse at   runtime.

Both `/assess` and `/chat` use the same retriever and grounding logic.

------------------------------------------------------------------------

## API Endpoints

### Health Check

GET `/health`

Returns:

``` json
{
  "status": "ok"
}
```

------------------------------------------------------------------------

### Clinical Assessment

POST `/assess`

Request:

``` json
{
  "patient_id": "PT-101"
}
```

Response includes:

-   category
-   rationale
-   recommended_action
-   supporting NG12 citations

------------------------------------------------------------------------

### Conversational Interface

POST `/chat`

Request:

``` json
{
  "patient_id": "PT-101",
  "message": "Do I need urgent referral?"
}
```

Response includes:

-   grounded answer
-   supporting citations
-   conversation history

------------------------------------------------------------------------

### View Conversation History

GET `/history/{patient_id}`

------------------------------------------------------------------------

### Clear Conversation History

DELETE `/history/{patient_id}`

------------------------------------------------------------------------

## Minimal Web UI

After starting the server:

http://127.0.0.1:8000/

Provides a simple interface to:

-   Run assessment
-   Ask follow-up chat questions
-   View citation-grounded responses

Swagger documentation:

http://127.0.0.1:8000/docs

------------------------------------------------------------------------

## Architecture

    app/
      agents/
        risk_assessor.py
        chat_agent.py
      rag/
        retriever.py
        vector_store.py
        ingest_pdf.py
      tools/
        patient_lookup.py
      memory/
        session_store.py
      main.py

    data/
      ng12.pdf
      index/
        faiss.index

------------------------------------------------------------------------

## Design Decisions

### Shared RAG Pipeline

Both endpoints reuse:

-   Same retriever
-   Same citation ranking logic
-   Same grounding rules
-   Same vector store

This ensures consistent reasoning and avoids re-embedding.

### Grounded Generation

The LLM:

-   Uses only retrieved NG12 excerpts
-   Avoids hallucinated thresholds or unsupported claims
-   Returns structured JSON responses
-   Falls back safely when insufficient evidence is retrieved

Fallback message:

"I couldn't find support in the NG12 text for that."

------------------------------------------------------------------------

## Installation

Clone the repository:

``` bash
git clone https://github.com/aarthijujjavarapu31-art/ng12-cancer-risk-assessor.git
cd ng12-cancer-risk-assessor
```

Create virtual environment:

``` bash
python -m venv .venv
```

Activate environment:

Windows:

    .venv\Scripts\activate

Mac/Linux:

    source .venv/bin/activate

Install dependencies:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Environment Variables

Create a `.env` file:

    GOOGLE_CLOUD_PROJECT=your-project-id
    GOOGLE_CLOUD_LOCATION=us-central1
    NG12_MODEL=gemini-2.0-flash-001
    NG12_TOP_K=10
    NG12_CHAT_TOP_CITATIONS=3

Do not commit credentials.

------------------------------------------------------------------------

## Authenticate Google Cloud

    gcloud auth application-default login
    gcloud config set project your-project-id
    gcloud auth application-default set-quota-project your-project-id

Ensure Vertex AI API is enabled.

------------------------------------------------------------------------

## Run Locally

``` bash
uvicorn app.main:app --reload
```

Server: http://127.0.0.1:8000\
Swagger: http://127.0.0.1:8000/docs

------------------------------------------------------------------------

## Run with Docker

Build image:

    docker build -t ng12-assessor .

Run container:

    docker run --rm -p 8000:8000 --env-file .env ng12-assessor

------------------------------------------------------------------------

## Alignment with Take-Home Requirements

-   FastAPI service
-   Dockerized configuration
-   Structured tool retrieval
-   Shared RAG pipeline across `/assess` and `/chat`
-   Citation-grounded responses
-   Multi-turn conversation memory
-   Prompt strategy documentation (PROMPTS.md, CHAT_PROMPTS.md)

------------------------------------------------------------------------

## Model Note

The brief referenced Gemini 1.5.\
This implementation uses the currently supported Vertex Gemini model
(configurable via `NG12_MODEL`) due to model lifecycle updates.
