import logging
import os
import time
from typing import Any, List, Optional, Sequence

from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

try:  # pragma: no cover - import guard for optional dependency
    from llama_index.llms.openai import OpenAI as LlamaOpenAI
except ImportError as exc:  # pragma: no cover - handled at runtime
    LlamaOpenAI = None  # type: ignore[assignment]
    _LLAMA_IMPORT_ERROR: Optional[Exception] = exc
else:
    _LLAMA_IMPORT_ERROR = None

from rag_client import rag_search

logger = logging.getLogger("rag-chat")


def _coerce_float(value: Any) -> Optional[float]:
    """Return a float if possible, otherwise None."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class RemoteRAGRetriever(BaseRetriever):
    """Retriever that proxies to the RAG service search endpoint."""

    def __init__(
        self,
        *,
        top_k: int,
        base_url: Optional[str] = None,
        postprocessors: Optional[Sequence[Any]] = None,
    ) -> None:
        super().__init__()
        self._top_k = top_k
        self._base_url = base_url
        self._session_id: Optional[str] = None
        self._postprocessors = [p for p in (postprocessors or []) if p is not None]

    def set_session(self, session_id: Optional[str]) -> None:
        self._session_id = session_id

    async def aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query = query_bundle.query_str.strip()
        if not query:
            return []

        started = time.perf_counter()
        try:
            payload = await rag_search(
                query,
                session_id=self._session_id,
                top_k=self._top_k,
                base_url=self._base_url,
            )
        except Exception as exc:  # pragma: no cover - network resilience
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.error(
                "Remote RAG request failed: %s",
                exc,
                extra={
                    "event": "rag_retrieve_error",
                    "session": self._session_id,
                    "elapsed_ms": round(elapsed_ms, 2),
                },
            )
            return []

        elapsed_ms = (time.perf_counter() - started) * 1000
        if not isinstance(payload, dict):
            logger.warning(
                "Unexpected RAG payload type: %s",
                type(payload).__name__,
                extra={
                    "event": "rag_retrieve_payload_mismatch",
                    "session": self._session_id,
                    "elapsed_ms": round(elapsed_ms, 2),
                },
            )
            return []

        raw_items = payload.get("items", [])
        if not isinstance(raw_items, list):
            logger.warning(
                "RAG payload missing item list",
                extra={
                    "event": "rag_retrieve_payload_empty",
                    "session": self._session_id,
                    "elapsed_ms": round(elapsed_ms, 2),
                },
            )
            return []

        nodes: List[NodeWithScore] = []

        for raw in raw_items:
            if not isinstance(raw, dict):
                continue

            text_candidates = [
                raw.get("text"),
                raw.get("content"),
                raw.get("chunk"),
            ]
            text = next((t.strip() for t in text_candidates if isinstance(t, str) and t.strip()), "")
            if not text:
                continue

            metadata: dict[str, Any] = {}
            for key in ("source", "title", "url", "id"):
                value = raw.get(key)
                if isinstance(value, str) and value.strip():
                    metadata[key] = value.strip()

            raw_metadata = raw.get("metadata")
            if isinstance(raw_metadata, dict):
                for key, value in raw_metadata.items():
                    if value is not None and key not in metadata:
                        metadata[key] = value

            distance = _coerce_float(metadata.get("distance"))
            if distance is None:
                distance = _coerce_float(raw.get("score"))
            if distance is None:
                distance = 0.0

            metadata["distance"] = distance

            score = -distance  # lower distance is better; NodeWithScore expects higher is better

            node = TextNode(text=text, metadata=metadata)
            nodes.append(NodeWithScore(node=node, score=score))

        if self._postprocessors:
            processed = nodes
            for processor in self._postprocessors:
                try:
                    processed = processor.postprocess_nodes(
                        processed,
                        query_bundle=query_bundle,
                    )
                except TypeError:
                    processed = processor.postprocess_nodes(  # type: ignore[arg-type]
                        processed,
                        query_bundle=query_bundle,
                    )
            nodes = processed

        logger.info(
            "Remote RAG retrieved %s nodes (raw_items=%s) in %.1fms",
            len(nodes),
            len(raw_items),
            elapsed_ms,
            extra={
                "event": "rag_retrieve_complete",
                "session": self._session_id,
                "hits": len(nodes),
                "requested_top_k": self._top_k,
                "elapsed_ms": round(elapsed_ms, 2),
            },
        )

        return nodes

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:  # pragma: no cover - sync API unused
        raise NotImplementedError("RemoteRAGRetriever supports async retrieval only")


class RAGChatEngine:
    """Thin wrapper around LlamaIndex chat engine that pulls context from rag_service."""

    def __init__(
        self,
        *,
        top_k: int,
        base_url: Optional[str] = None,
        rerankers: Optional[Sequence] = None,
        chat_model: Optional[str] = None,
        temperature: Optional[float] = None,
        memory_token_limit: Optional[int] = None,
    ) -> None:
        resolved_base_url = base_url or os.getenv("RAG_BASE_URL")
        postprocessors = [r for r in (rerankers or []) if r is not None]

        self._retriever = RemoteRAGRetriever(
            top_k=top_k,
            base_url=resolved_base_url,
            postprocessors=postprocessors,
        )

        if LlamaOpenAI is None:  # pragma: no cover - environment issue
            raise RuntimeError(
                "llama-index-llms-openai is unavailable; ensure compatible versions are installed."
            ) from _LLAMA_IMPORT_ERROR

        model_name = chat_model or os.getenv("RAG_CHAT_MODEL", "gpt-5-mini")
        model_temperature = (
            temperature
            if temperature is not None
            else float(os.getenv("RAG_CHAT_TEMPERATURE", "0.2"))
        )

        self._llm = LlamaOpenAI(model=model_name, temperature=model_temperature)

        postprocessors = [r for r in (rerankers or []) if r is not None]

        response_synthesizer = get_response_synthesizer(
            llm=self._llm,
            response_mode=os.getenv("RAG_RESPONSE_MODE", "compact"),
        )

        memory_limit = memory_token_limit or int(
            os.getenv("RAG_CHAT_MEMORY_TOKENS", "4096")
        )

        query_engine = RetrieverQueryEngine(
            retriever=self._retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=postprocessors or None,
        )

        self._chat_engine = CondenseQuestionChatEngine.from_defaults(
            query_engine=query_engine,
            memory=ChatMemoryBuffer.from_defaults(token_limit=memory_limit),
            llm=self._llm,
        )

    def reset(self) -> None:
        """Clear accumulated chat memory."""
        self._chat_engine.reset()

    async def aquery(self, question: str, *, session_id: Optional[str] = None) -> str:
        question = (question or "").strip()
        if not question:
            return ""

        self._retriever.set_session(session_id)
        started = time.perf_counter()
        try:
            response = await self._chat_engine.achat(question)
        except Exception as exc:  # pragma: no cover - defensive
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.error(
                "RAG chat engine failed: %s",
                exc,
                exc_info=True,
                extra={
                    "event": "rag_chat_error",
                    "session": session_id,
                    "elapsed_ms": round(elapsed_ms, 2),
                },
            )
            return "Entschuldigung, ich konnte keine weiteren Informationen finden."

        elapsed_ms = (time.perf_counter() - started) * 1000

        text = getattr(response, "response", None)
        if not isinstance(text, str) or not text.strip():
            text = getattr(response, "message", None)
        if not isinstance(text, str) or not text.strip():
            text = getattr(response, "text", None)

        if isinstance(text, str) and text.strip():
            answer = text.strip()
        else:
            answer = str(response)

        logger.info(
            "RAG chat completed in %.1fms",
            elapsed_ms,
            extra={
                "event": "rag_chat_complete",
                "session": session_id,
                "elapsed_ms": round(elapsed_ms, 2),
            },
        )

        return answer
