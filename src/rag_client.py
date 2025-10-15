import os
import time
import logging
import aiohttp
import asyncio
from typing import Optional, Dict, Any, Tuple

log = logging.getLogger("rag-client")
BASE_URL = os.getenv("RAG_BASE_URL", "http://rag:8000")  # default container service
TOP_K = int(os.getenv("RAG_TOP_K", "6"))
MAX_RETRIES = 1
RETRY_DELAY = float(os.getenv("RAG_RETRY_DELAY", "1.0"))

async def _make_rag_request(url: str, payload: Dict[str, Any], attempt: int = 1) -> Tuple[Optional[Dict[str, Any]], bool]:
    """
    Make a single RAG request with timeout and error handling.
    Returns (result, retryable). If result is None and retryable is False, the caller should stop retrying.
    """
    try:
        timeout = aiohttp.ClientTimeout(total=8)
        log.debug(f"RAG request attempt {attempt}: {url} with payload {payload}")

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    log.debug(f"RAG request successful: status {response.status}")
                    return data, False

                response_text = await response.text()
                log.warning(f"RAG search failed (attempt {attempt}): {url} -> {response.status}: {response_text}")

                # Do not retry on client errors (4xx)
                if 400 <= response.status < 500:
                    log.error(f"Client error, not retrying: {response.status}")
                    return None, False

                # Retry on 5xx and other transient errors
                return None, True

    except aiohttp.ClientError as e:
        log.warning(f"RAG network error (attempt {attempt}): {e}")
        return None, True
    except asyncio.TimeoutError:
        log.warning(f"RAG timeout (attempt {attempt}): request took too long")
        return None, True
    except Exception as e:
        log.error(f"RAG unexpected error (attempt {attempt}): {e}")
        return None, True

async def rag_search(
    question: str,
    *,
    session_id: Optional[str] = None,
    top_k: Optional[int] = None,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Call the RAG service /search endpoint and return its raw payload."""

    question = (question or "").strip()
    if not question:
        log.warning("Empty question provided to RAG, skipping request")
        return {}

    search_url = f"{(base_url or BASE_URL).rstrip('/')}/api/rag/search"
    payload: Dict[str, Any] = {"question": question, "top_k": top_k or TOP_K}
    if session_id:
        payload["session_id"] = session_id

    log.info(
        "RAG search request: question='%s' session_id=%s top_k=%s base_url=%s",
        f"{question[:50]}..." if len(question) > 50 else question,
        session_id,
        payload["top_k"],
        base_url or BASE_URL,
        extra={
            "event": "rag_http_start",
            "session": session_id,
            "top_k": payload["top_k"],
        },
    )

    started = time.perf_counter()

    for attempt in range(1, MAX_RETRIES + 1):
        result, retryable = await _make_rag_request(search_url, payload, attempt)

        if result is not None:
            if isinstance(result, dict):
                elapsed_ms = (time.perf_counter() - started) * 1000
                log.info(
                    "RAG search success: %d items, formattedContext=%s",
                    len(result.get("items", []) or []),
                    "yes" if result.get("formattedContext") else "no",
                    extra={
                        "event": "rag_http_complete",
                        "session": session_id,
                        "elapsed_ms": round(elapsed_ms, 2),
                        "hits": len(result.get("items", []) or []),
                    },
                )
                return result

            log.warning("RAG returned non-dict payload: %s", result)
            return {}

        if not retryable:
            break

        if attempt < MAX_RETRIES:
            log.info("RAG attempt %d failed, retrying in %.1fs", attempt, RETRY_DELAY)
            await asyncio.sleep(RETRY_DELAY)
        else:
            elapsed_ms = (time.perf_counter() - started) * 1000
            log.error(
                "RAG failed after %d attempts",
                MAX_RETRIES,
                extra={
                    "event": "rag_http_failed",
                    "session": session_id,
                    "elapsed_ms": round(elapsed_ms, 2),
                },
            )

    log.error("RAG service unavailable, returning empty result")
    return {}


async def retrieve_context(question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Backwards-compatible helper that returns {'formattedContext': str, 'items': [...]}.
    """

    result = await rag_search(question, session_id=session_id)
    if not result:
        return {}

    formatted_context = result.get("formattedContext", "") if isinstance(result, dict) else ""
    items = result.get("items", []) if isinstance(result, dict) else []
    return {"formattedContext": formatted_context, "items": items}
