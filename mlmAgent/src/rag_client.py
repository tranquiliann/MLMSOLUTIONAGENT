import os
import logging
import aiohttp

log = logging.getLogger("rag-client")
BASE_URL = os.getenv("RAG_BASE_URL", "http://rag:8000")  # container-to-container
TOP_K = int(os.getenv("RAG_TOP_K", "6"))

async def retrieve_context(question: str, session_id: str | None = None) -> dict:
    """Call Repo C's RAG service and return {'formattedContext': str, 'items': [...]}.
    Returns {} on failure (agent will fall back to default behavior).
    """
    url = f"{BASE_URL.rstrip('/')}/api/rag/search"
    payload = {"question": question, "top_k": TOP_K}
    try:
        timeout = aiohttp.ClientTimeout(total=8)
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.post(url, json=payload) as resp:
                if resp.status != 200:
                    txt = await resp.text()
                    log.warning("RAG search %s -> %s: %s", url, resp.status, txt)
                    return {}
                data = await resp.json()
                # Expected: { topK, items, formattedContext }
                return {
                    "formattedContext": data.get("formattedContext", ""),
                    "items": data.get("items", []),
                }
    except Exception as e:
        log.exception("RAG request failed: %s", e)
        return {}
