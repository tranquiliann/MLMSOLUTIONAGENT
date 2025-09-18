import os
import logging
import aiohttp
import asyncio
from typing import Optional, Dict, Any

log = logging.getLogger("rag-client")
BASE_URL = os.getenv("RAG_BASE_URL", "http://rag:8000")  # container-to-container
TOP_K = int(os.getenv("RAG_TOP_K", "6"))
MAX_RETRIES = int(os.getenv("RAG_MAX_RETRIES", "3"))
RETRY_DELAY = float(os.getenv("RAG_RETRY_DELAY", "1.0"))

async def _make_rag_request(url: str, payload: Dict[str, Any], attempt: int = 1) -> Optional[Dict[str, Any]]:
    """Make a single RAG request with timeout and error handling."""
    try:
        timeout = aiohttp.ClientTimeout(total=8)
        log.debug(f"RAG request attempt {attempt}: {url} with payload {payload}")

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    log.debug(f"RAG request successful: status {response.status}")
                    return data
                else:
                    response_text = await response.text()
                    log.warning(f"RAG search failed (attempt {attempt}): {url} -> {response.status}: {response_text}")

                    # Don't retry on client errors (4xx)
                    if 400 <= response.status < 500:
                        log.error(f"Client error, not retrying: {response.status}")
                        return None

                    return None

    except aiohttp.ClientError as e:
        log.warning(f"RAG network error (attempt {attempt}): {e}")
        return None
    except asyncio.TimeoutError:
        log.warning(f"RAG timeout (attempt {attempt}): request took too long")
        return None
    except Exception as e:
        log.error(f"RAG unexpected error (attempt {attempt}): {e}")
        return None

async def retrieve_context(question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """Call RAG service with retry logic and return {'formattedContext': str, 'items': [...]}.

    Returns empty dict on all failures (agent falls back to default behavior).
    """
    if not question or not question.strip():
        log.warning("Empty question provided to RAG, skipping request")
        return {}

    url = f"{BASE_URL.rstrip('/')}/api/rag/search"
    payload = {"question": question, "top_k": TOP_K}

    log.info(f"RAG request: question='{question[:50]}...' session_id={session_id}")

    for attempt in range(1, MAX_RETRIES + 1):
        result = await _make_rag_request(url, payload, attempt)

        if result is not None:
            # Validate response structure
            if isinstance(result, dict) and ("formattedContext" in result or "items" in result):
                formatted_context = result.get("formattedContext", "")
                items = result.get("items", [])

                log.info(f"RAG success: {len(items)} items, context length: {len(formatted_context)}")

                return {
                    "formattedContext": formatted_context,
                    "items": items,
                }
            else:
                log.warning(f"RAG returned invalid response structure: {result}")

        # If this wasn't the last attempt, wait before retrying
        if attempt < MAX_RETRIES:
            log.info(f"RAG attempt {attempt} failed, retrying in {RETRY_DELAY}s...")
            await asyncio.sleep(RETRY_DELAY)
        else:
            log.error(f"RAG failed after {MAX_RETRIES} attempts")

    # All retries exhausted
    log.error("RAG service unavailable, agent will continue without context")
    return {}
