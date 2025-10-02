import logging
import os
import time
from typing import Optional

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from livekit.agents import (
    NOT_GIVEN,
    Agent,
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

try:  # Compatibility with older LiveKit Agents builds
    from livekit.agents import AgentFalseInterruptionEvent  # type: ignore
except ImportError:  # pragma: no cover - legacy versions
    AgentFalseInterruptionEvent = None  # type: ignore
from livekit.agents.llm import function_tool
from livekit.plugins import cartesia, deepgram, noise_cancellation, openai, silero
from livekit.plugins.turn_detector.models import HG_MODEL, MODEL_REVISIONS, ONNX_FILENAME
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from health_server import start_health_server, stop_health_server
from rag_client import BASE_URL, retrieve_context  # RAG retrieval via your RAG service
from rag_chat import RAGChatEngine

#
# LlamaIndex reranker removed - not used in basic setup
#

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
    if os.getenv("SKIP_TURN_DETECTOR_CHECK", "false").lower() in {"1", "true", "yes"}:
        logger.warning("Skipping turn detector asset validation due to env override")
        return

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

# -----------------------------
# Deutschsprachige System-Policy
# -----------------------------
BASE_POLICY = """
Du bist Fibi, die freundliche digitale Coachin von BVNM für die F3-Methode erschaffen von lumea.AI

# Nutzung von RAG
- Wenn du fachliche Informationen aus dem BVNM-Wissensarchiv brauchst, rufe das
  Funktionstool `query_rag` mit einer klar formulierten Frage auf.
- Verwende das Tool nicht für Smalltalk oder wenn du die Antwort bereits sicher
  weißt.
- Warte auf die Tool-Antwort und integriere die relevanten Fakten natürlich in
  deine Antwort, ohne das Tool oder interne Mechanismen zu erwähnen.

# Stil & Sprache
- Antworte kurz, klar und hilfreich. Stelle höchstens eine Rückfrage zugleich.
- Spiegle die Nutzersprache; Standard ist Deutsch (BVNM).
- Nenne keine internen Systemdetails (RAG, VAD, Modelle, Schlüssel etc.).

"""


# -----------------------------
# Health check function (RAG)
# -----------------------------
AGENT_SERVICE_NAME = os.getenv("AGENT_SERVICE_NAME", "mlmsolution-agent")
AGENT_VERSION = os.getenv("AGENT_VERSION", "dev")
HEALTH_HOST = os.getenv("AGENT_HEALTH_HOST", "0.0.0.0")
HEALTH_PORT = int(os.getenv("AGENT_HEALTH_PORT", "8081"))


async def health_check() -> dict:
    """Perform health check for agent and RAG connectivity."""
    health_status = {
        "status": "healthy",
        "service": AGENT_SERVICE_NAME,
        "version": AGENT_VERSION,
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
        health_status["checks"]["rag"]["details"]["base_url"] = os.getenv("RAG_BASE_URL", "")
    except Exception as e:
        logger.error(f"RAG health check failed: {e}")
        health_status["checks"]["rag"]["status"] = "unhealthy"
        health_status["checks"]["rag"]["details"]["error"] = str(e)
        health_status["status"] = "unhealthy"

    return health_status


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=BASE_POLICY)
        self._rag_engine = RAGChatEngine(
            top_k=6,
            base_url=BASE_URL,
            rerankers=None,
        )
        self._current_session_id: Optional[str] = None

    def _get_session_id(self, context: RunContext) -> str:
        session = getattr(context, "session", None) or getattr(self, "session", None)
        room = getattr(session, "room", None)
        if room and getattr(room, "name", None):
            return room.name
        return "unknown_session"

    @function_tool
    async def query_rag(self, context: RunContext, question: str) -> str:
        """Fragt das Wissensarchiv nach zusätzlichen Fakten."""
        session_id = self._get_session_id(context)
        if session_id != self._current_session_id:
            logger.debug("Starting new RAG conversation for session %s", session_id)
            self._rag_engine.reset()
            self._current_session_id = session_id

        normalized_question = (question or "").strip()
        if not normalized_question:
            logger.info(
                "RAG tool received empty question",
                extra={
                    "event": "rag_tool_empty",
                    "session": session_id,
                },
            )
            return "Ich habe dazu leider keine weiteren Informationen gefunden."

        try:
            logger.info(
                "RAG tool invocation: session=%s question='%s'",
                session_id,
                normalized_question[:80],
                extra={
                    "event": "rag_tool_start",
                    "session": session_id,
                },
            )
            started = time.perf_counter()
            answer = await self._rag_engine.aquery(normalized_question, session_id=session_id)
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.info(
                "RAG tool completed in %.1fms",
                elapsed_ms,
                extra={
                    "event": "rag_tool_complete",
                    "session": session_id,
                    "elapsed_ms": round(elapsed_ms, 2),
                    "answer_empty": not bool(answer and answer.strip()),
                },
            )
            return answer or "Ich habe dazu leider keine weiteren Informationen gefunden."
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("RAG tool failed: %s", exc, exc_info=True)
            return "Entschuldigung, ich konnte dazu gerade keine Wissensbasis finden."

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

    health_server = None
    try:
        health_server = await start_health_server(
            status_provider=health_check,
            host=HEALTH_HOST,
            port=HEALTH_PORT,
        )
        if health_server is not None:
            logger.info(
                "Health endpoint listening on %s:%s",
                HEALTH_HOST,
                HEALTH_PORT,
                extra={"event": "health_server_started"},
            )
        else:
            logger.warning(
                "Health endpoint disabled (bind failed)",
                extra={"event": "health_server_unavailable"},
            )
    except Exception as exc:  # pragma: no cover - startup guard
        logger.error(
            "Failed to start health endpoint: %s",
            exc,
            exc_info=True,
            extra={"event": "health_server_error"},
        )

    session = AgentSession(
        llm=openai.LLM(model="gpt-4o-mini"),
        stt=deepgram.STT(model="nova-3", language="multi"),
        tts=openai.TTS(voice="alloy"),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
    )

    if AgentFalseInterruptionEvent is not None:

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

    if health_server is not None:
        async def _stop_health() -> None:
            await stop_health_server(health_server)

        ctx.add_shutdown_callback(_stop_health)

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
