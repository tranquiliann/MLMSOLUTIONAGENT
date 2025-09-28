# Unified RAG Flow

This document summarises how the agent, retrieval middleware, and infra services collaborate after the RAG refactor.

## High-level flow

```
User ↴
LLM (Chat session)
  │
  ├─> Function tool: query_rag(context, question)
  │       │
  │       ├─> RAGChatEngine (LlamaIndex)
  │       │       │
  │       │       ├─> RemoteRAGRetriever.aretrieve → rag_client.rag_search
  │       │       │       │
  │       │       │       ├─HTTP POST /api/rag/search (rag_service)
  │       │       │       │       │
  │       │       │       │       └─> Redis (vector index)
  │       │       │       │
  │       │       ├─> (optional) FlagEmbeddingReranker
  │       │       └─> Response synthesis (LLM)
  │       │
  │       └─> Structured answer returned to tool
  │
  └─> Assistant merges answer into final reply for the user
```

## Components

- **query_rag tool (`agent.Assistant`)** – The LLM decides when to call this tool. It tracks `session_id`s, resets chat memory between rooms, and emits structured logs (`event=rag_tool_*`).
- **`RAGChatEngine` (agent)** – Wraps LlamaIndex’ `CondenseQuestionChatEngine`, optionally applying a `FlagEmbeddingReranker`. The engine defends against malformed payloads, records latency, and normalises metadata (`distance`, `source`, etc.).
- **`RemoteRAGRetriever` (agent)** – Proxy retriever that issues `rag_search` HTTP requests. It tolerates partial responses, converts Redis distances to LlamaIndex scores, and logs timing & hit counts.
- **`rag_client.rag_search` (agent)** – Thin HTTP client that respects the `RAG_BASE_URL` environment variable, retries once, and emits structured success/failure logs.
- **`rag_service` (infra)** – FastAPI service exposing `/api/rag/search`. Responses now include a consistent payload:
  ```json
  {
    "items": [
      {
        "text": "…",
        "content": "…",
        "score": 0.12,
        "id": "doc:…",
        "source": "…",
        "title": "…",
        "url": "…",
        "metadata": {"distance": 0.12, "source": "…"}
      }
    ],
    "formattedContext": "…"
  }
  ```
  Additional `/health` and `/version` endpoints ease Kubernetes probes.

## Observability

Key structured log events:

- `rag_tool_start`, `rag_tool_complete`, `rag_tool_empty`
- `rag_retrieve_complete`, `rag_retrieve_error`, `rag_retrieve_payload_*`
- `rag_chat_complete`, `rag_chat_error`
- `rag_http_start`, `rag_http_complete`, `rag_http_failed`
- `rag_service_search_start`, `rag_service_search_complete`
- `health_server_started`, `health_server_unavailable`

Every event captures hits, session identifiers, and elapsed milliseconds where applicable.

## Developer workflow

- Install tooling: `make install`
- Run automated checks: `make test` / `make lint`
- Launch the agent: `make run-agent`
- Start the retrieval service (Docker): `make run-infra`

A manual end-to-end probe is available via `python scripts/smoke_query_rag.py` once the infra stack is reachable.

## Tests

New unit & integration tests cover:

1. `RemoteRAGRetriever.aretrieve` mapping JSON payloads to `NodeWithScore` objects.
2. `Assistant.query_rag` returning grounded text when the RAG service (mocked) yields a single hit.

Running `pytest -q` (with `PYTHONPATH=src`) exercises both cases.
