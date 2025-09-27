import logging
import os
import asyncio

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from livekit.agents import (
    NOT_GIVEN,
    Agent,
    AgentFalseInterruptionEvent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.llm import function_tool
from livekit.plugins import cartesia, deepgram, noise_cancellation, openai, silero
from livekit.plugins.turn_detector.models import HG_MODEL, MODEL_REVISIONS, ONNX_FILENAME
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from .rag_client import retrieve_context  # RAG retrieval via your RAG service

#
# LlamaIndex reranker (local, no external LLM calls)
#
from llama_index.core.schema import TextNode, NodeWithScore, QueryBundle  # type: ignore
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker  # type: ignore

logger = logging.getLogger("agent")

load_dotenv(".env.local")


def validate_environment() -> None:
    """Validate required environment variables at startup."""
    required_vars = {
        "RAG_BASE_URL": "URL for RAG service connectivity",
        "LIVEKIT_URL": "LiveKit server URL",
        "LIVEKIT_API_KEY": "LiveKit API key for authentication",
        "LIVEKIT_API_SECRET": "LiveKit API secret for authentication",
    }

    missing_vars = []
    for var, description in required_vars.items():
        value = os.getenv(var)
        if not value or not value.strip():
            missing_vars.append(f"{var} ({description})")

    if missing_vars:
        error_msg = "Missing required environment variables:\n" + "\n".join(f"  - {var}" for var in missing_vars)
        logger.error(error_msg)
        raise RuntimeError(f"Environment validation failed:\n{error_msg}")

    logger.info("Environment validation passed - all required variables are set")


# Validate environment on module import
validate_environment()


def ensure_turn_detector_assets() -> None:
    """Fail fast if multilingual VAD assets are missing."""
    multilingual_revision = MODEL_REVISIONS["multilingual"]
    required_assets = [
        ("languages.json", {}),
        ("tokenizer.json", {}),
        ("tokenizer_config.json", {}),
        ("special_tokens_map.json", {}),
        ("vocab.json", {}),
        ("merges.txt", {}),
        (ONNX_FILENAME, {"subfolder": "onnx"}),
    ]

    for filename, kwargs in required_assets:
        try:
            hf_hub_download(
                repo_id=HG_MODEL,
                filename=filename,
                revision=multilingual_revision,
                local_files_only=True,
                **kwargs,
            )
        except Exception as exc:  # pragma: no cover - fail fast in production
            location = f"{kwargs['subfolder']}/{filename}" if "subfolder" in kwargs else filename
            raise RuntimeError(
                "Missing multilingual turn detector asset. "
                f"Ensure {location} is pre-downloaded for revision {multilingual_revision}."
            ) from exc


ensure_turn_detector_assets()

#
# -----------------------------
# Local reranker configuration
# -----------------------------
#
# Defaults are safe; override via .env.local if you want.
RERANK_MODEL = os.getenv("RAG_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_TOP_N = int(os.getenv("RAG_RERANK_TOP_N", "6"))
RERANK_CAP   = int(os.getenv("RAG_RERANK_CAP", "24"))   # candidates to re-rank (latency control)
RERANK_FP16  = os.getenv("RAG_RERANK_FP16", "false").lower() == "true"  # set true if you run CUDA

try:
    _flag_reranker = FlagEmbeddingReranker(
        model=RERANK_MODEL,
        top_n=RERANK_TOP_N,
        use_fp16=RERANK_FP16,
    )
    logger.info(
        f"Loaded local reranker: {RERANK_MODEL} (top_n={RERANK_TOP_N}, cap={RERANK_CAP}, fp16={RERANK_FP16})"
    )
except Exception as e:
    logger.warning(f"FlagEmbeddingReranker init failed; disabling rerank. Error: {e}")
    _flag_reranker = None

def _rerank_items_with_bge(items: list, query: str) -> list:
    """Reorder items locally using the BGE cross-encoder; keep your item shape."""
    if not items or _flag_reranker is None:
        return items
    # Cap candidates to bound latency
    candidates = items[:RERANK_CAP]

    nodes = []
    for it in candidates:
        txt = (it.get("text") or it.get("content") or it.get("chunk") or "").strip()
        if not txt:
            continue
        node = TextNode(text=txt, metadata=it.get("metadata", {}))
        nodes.append(NodeWithScore(node=node, score=float(it.get("score", 0.0))))

    if not nodes:
        return items

    out_nodes = _flag_reranker.postprocess_nodes(
        nodes, query_bundle=QueryBundle(query_str=query)
    )
    # Map back to your item dict shape (text + score + metadata)
    remapped = []
    for n in out_nodes:
        remapped.append({
            "text": n.node.get_content(),
            "score": float(n.score or 0.0),
            "metadata": dict(n.node.metadata or {}),
        })
    return remapped


# -----------------------------
# Deutschsprachige System-Policy
# -----------------------------
BASE_POLICY = """
Du bist Fibi, die freundliche digitale Coachin von BVNM für die F3-Methode erschaffen von lumea.AI

# Nutzung von RAG
- Falls im Chatverlauf VOR der Nutzeranfrage eine Assistenz-Nachricht mit
  'RAG_CONTEXT_BEGIN' ... 'RAG_CONTEXT_END' vorhanden ist, nutze deren Inhalt
  als sachliche Hintergrundinformation. Integriere relevante Fakten natürlich
  in deine Antwort, ohne RAG, Snippets oder Mechanismen zu erwähnen.
- Falls KEINE solche RAG-Assistenz-Nachricht vorhanden ist, antworte ganz normal
  aus deinem allgemeinen Können heraus. 
- Falls RAG-Kontext vorhanden, aber offenkundig themenfremd zur Nutzerabsicht ist,
  ignoriere ihn und antworte normal.

# Stil & Sprache
- Antworte kurz, klar und hilfreich. Stelle höchstens eine Rückfrage zugleich.
- Spiegle die Nutzersprache; Standard ist Deutsch (BVNM).
- Nenne keine internen Systemdetails (RAG, VAD, Modelle, Schlüssel etc.).

"""


# -----------------------------
# Health check function (RAG)
# -----------------------------
async def health_check() -> dict:
    """Perform health check for agent and RAG connectivity."""
    health_status = {
        "status": "healthy",
        "checks": {
            "environment": {"status": "healthy", "details": {}},
            "rag": {"status": "unknown", "details": {}},
        },
    }

    required_env_vars = ["RAG_BASE_URL", "LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"]
    for var in required_env_vars:
        value = os.getenv(var)
        if value:
            health_status["checks"]["environment"]["details"][var] = "set"
        else:
            health_status["checks"]["environment"]["details"][var] = "missing"
            health_status["checks"]["environment"]["status"] = "unhealthy"
            health_status["status"] = "unhealthy"

    try:
        rag_result = await retrieve_context("health check", "health_check")
        has_items = bool(isinstance(rag_result, dict) and rag_result.get("items"))
        has_context = bool(isinstance(rag_result, dict) and (rag_result.get("formattedContext") or "").strip())

        if has_items or has_context:
            health_status["checks"]["rag"]["status"] = "healthy"
            health_status["checks"]["rag"]["details"]["response"] = "ok"
        else:
            # Leer ist kein harter Fehler -> degraded
            health_status["checks"]["rag"]["status"] = "degraded"
            health_status["checks"]["rag"]["details"]["response"] = "empty"
    except Exception as e:
        logger.error(f"RAG health check failed: {e}")
        health_status["checks"]["rag"]["status"] = "unhealthy"
        health_status["checks"]["rag"]["details"]["error"] = str(e)
        health_status["status"] = "unhealthy"

    return health_status


def _rag_is_useful(rag: dict) -> bool:
    """
    Relevanzkriterium ohne hartkodierte Smalltalk-Listen:
    - Nützlich, wenn mindestens ein Item existiert (optional mit brauchbarem Score),
      ODER der formatierte Kontext substanzielle Länge hat (>= 200 Zeichen).
    """
    if not rag or not isinstance(rag, dict):
        return False

    items = rag.get("items") or []
    if items:
        try:
            max_score = max(
                float(i.get("score") or i.get("relevance") or i.get("similarity") or 0.0)
                for i in items
                if isinstance(i, dict)
            )
            if max_score >= 0.5:
                return True
        except Exception:
            # Falls Scores fehlen/uneinheitlich sind, genügt die Existenz von Items.
            return True
        return True

    ctx = (rag.get("formattedContext") or "").strip()
    return len(ctx) >= 200


async def _condense_question_with_budget(session_llm, question: str, history_text: str = "", ms_budget: int = 150) -> str:
    """
    Versucht, die Nutzerfrage binnen ms_budget zu einer eigenständigen Frage umzuformulieren.
    Bei Timeout/Fehler wird die Originalfrage zurückgegeben (kein Latenzaufschlag über Budget).
    """
    question = (question or "").strip()
    if not question:
        return ""

    sys_msg = (
        "Du bist ein Umschreiber. Formuliere die Nutzerfrage so um, dass sie ohne Kontext eindeutig ist. "
        "Übernimm relevante Entitäten/Orte/Zeiträume aus dem Verlauf. "
        "Antworte NUR mit der umgeschriebenen Frage."
    )
    user_payload = f"Verlauf:\n{(history_text or '').strip()}\n\nFrage:\n{question}\n\nUmschreiben:"

    async def _call():
        resp = await session_llm.chat(messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_payload},
        ])
        text = getattr(resp, "text", None) or getattr(resp, "message", None) or ""
        return (text or question).strip()

    try:
        return await asyncio.wait_for(_call(), timeout=ms_budget / 1000.0)
    except Exception:
        return question


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=BASE_POLICY)
        self._last_user_utterance: str = ""   # für Folgefragen-Umschreibung

    # Läuft NACH der Nutzeräußerung und VOR der Modellantwort.
    # RAG wird nur als assistant-Nachricht injiziert, wenn nützlich.
    # Keine system-Nachrichten pro Turn.
    async def on_user_turn_completed(self, turn_ctx, new_message):
        try:
            text = (getattr(new_message, "text_content", None) or getattr(new_message, "text", ""))
            if not text:
                return
            text = text.strip()
            if not text:
                return

            # Session-ID (Raumname bevorzugt) für Tracing
            session_id = getattr(turn_ctx, "room", None)
            if session_id and hasattr(session_id, "name"):
                session_id = session_id.name
            else:
                session_id = "unknown_session"

            # 1) kurze Historie (hier nur die letzte Nutzerfrage; bewusst minimal für Latenz)
            history_text = self._last_user_utterance

            # 2) zeitbudgetierte Umschreibung; bei Timeout/Fehler wird einfach `text` genutzt
            llm = getattr(getattr(self, "session", None), "llm", None) or getattr(getattr(turn_ctx, "session", None), "llm", None)
            condensed = await _condense_question_with_budget(llm or self.session.llm, text, history_text, ms_budget=150)

            # 3) RAG mit der kondensierten (oder Original‑)Frage
            rag = await retrieve_context(question=condensed, session_id=session_id)

            # 4) letzte Nutzeräußerung für den nächsten Turn merken
            self._last_user_utterance = text

            # Nur injizieren, wenn nützlich
            if _rag_is_useful(rag):
                items = rag.get("items") or []
                formatted = (rag.get("formattedContext") or "").strip()

                # Wenn Items vorhanden sind, lokal mit BGE neu reihen und daraus Kontext bauen
                if items:
                    items = _rerank_items_with_bge(items, query=condensed)
                    top_blocks = [i.get("text", "").strip() for i in items if i.get("text")]
                    # Baue kompakten Kontext aus den Top-N; vermeide unnötig lange Injektion
                    formatted_from_items = "\n\n".join(top_blocks[:RERANK_TOP_N]).strip()
                    if formatted_from_items:
                        formatted = formatted_from_items

                # after building formatted_from_items
                if not formatted and not formatted_from_items:
                    return  # nothing meaningful to inject

                if formatted:
                    turn_ctx.add_message(
                        role="assistant",
                        content=f"RAG_CONTEXT_BEGIN\n{formatted}\nRAG_CONTEXT_END",
                    )

            # Wenn nicht nützlich: keine Injektion → Modell antwortet normal gemäß BASE_POLICY

        except Exception as e:
            logger.exception(f"pre-reply RAG injection failed (continuing without RAG): {e}")
            # Fehler blockiert den Turn nicht; wir injizieren nichts.

    # System-Health-Tool (intern); nie für Nutzerwissen
    @function_tool
    async def check_system_health(self, context: RunContext):
        """Check the health status of the agent system including RAG connectivity and environment configuration."""
        try:
            health_result = await health_check()
            status = health_result["status"]

            if status == "healthy":
                summary = "System is healthy. All checks passed."
            elif status == "unhealthy":
                summary = "System has issues that need attention."
            else:
                summary = "System is partially degraded."

            # Plain text, no emojis
            details = []
            for check_name, check_data in health_result["checks"].items():
                details.append(f"{check_name}: {check_data['status']}")
                if check_data["details"]:
                    for key, value in check_data["details"].items():
                        details.append(f"  - {key}: {value}")

            return summary + "\n\n" + "\n".join(details)

        except Exception as e:
            logger.error(f"Health check tool failed: {e}")
            return f"Health check failed: {str(e)}"


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session = AgentSession(
        llm=openai.LLM(model="gpt-5-mini"),
        stt=deepgram.STT(model="nova-3", language="multi"),
        tts=openai.TTS(voice="alloy"),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=False,  # keep pre-reply RAG injection strict (assistant-role only)
    )

    @session.on("agent_false_interruption")
    def _on_agent_false_interruption(ev: AgentFalseInterruptionEvent):
        logger.info("false positive interruption, resuming")
        session.generate_reply(instructions=ev.extra_instructions or NOT_GIVEN)

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()

    # Erste Antwort: nur Startinstruktion auf Deutsch; BASE_POLICY ist bereits gesetzt.
    session.generate_reply(
        instructions=(
            "Starte das Gespräch jetzt mit einer freundlichen Begrüßung. Stelle dich genau einmal "
            "als Fibi vor, erinnere kurz an die drei Schritte (Registrieren, Aktivieren, Teilen) "
            "und frage anschließend, wobei du helfen kannst."
        ),
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
