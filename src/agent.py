import logging
import os
import time
from typing import Optional

DEFAULT_SERVICE_ENV = {
    "LIVEKIT_URL": os.getenv("LIVEKIT_URL", "wss://your-livekit-host"),
    "LIVEKIT_API_KEY": os.getenv("LIVEKIT_API_KEY", "YOUR_LIVEKIT_KEY"),
    "LIVEKIT_API_SECRET": os.getenv("LIVEKIT_API_SECRET", "YOUR_LIVEKIT_SECRET"),
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "sk-your-openai-key"),
    "DEEPGRAM_API_KEY": os.getenv("DEEPGRAM_API_KEY", "your-deepgram-key"),
    "RAG_BASE_URL": os.getenv("RAG_BASE_URL", "http://rag:8000"),
}

for _env_key, _env_value in DEFAULT_SERVICE_ENV.items():
    os.environ.setdefault(_env_key, _env_value)

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    metrics,
    MetricsCollectedEvent,
)
from livekit.agents.llm import function_tool
from livekit.plugins import google

from health_server import start_health_server, stop_health_server
from rag_client import BASE_URL, retrieve_context  # RAG retrieval via your RAG service
from rag_chat import RAGChatEngine

#
# LlamaIndex reranker removed - not used in basic setup
#

logger = logging.getLogger("agent")

load_dotenv(".env.local")

for _env_key, _env_value in DEFAULT_SERVICE_ENV.items():
    os.environ.setdefault(_env_key, _env_value)


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
        super().__init__(
            instructions=BASE_POLICY,
        )
        self._rag_engine = RAGChatEngine(
            top_k=6,
            base_url=BASE_URL,
            rerankers=None,
        )
        self._current_session_id: Optional[str] = None

    @function_tool
    async def query_rag(self, question: str) -> str:
        """Fragt das Wissensarchiv nach zusätzlichen Fakten."""
        session_id = "agent_session"
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
    async def check_system_health(self):
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

    await ctx.connect()

    session = AgentSession(
        llm=google.realtime.RealtimeModel(
            model="models/gemini-2.5-flash",
            instructions=BASE_POLICY,
            voice="Aoede"
        ),
    )

    agent = Assistant()

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
        agent=agent,
        room=ctx.room,
    )

    # Erste Antwort: nur Startinstruktion auf Deutsch; BASE_POLICY ist bereits gesetzt.
    # We await generate_reply since session handles output natively
    await session.generate_reply(
        instructions=(
            "Starte das Gespräch jetzt mit einer freundlichen Begrüßung. Stelle dich genau einmal "
            "als Fibi vor, erinnere kurz an die drei Schritte (Registrieren, Aktivieren, Teilen) "
            "und frage anschließend, wobei du helfen kannst."
        )
    )

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            ws_url=DEFAULT_SERVICE_ENV["LIVEKIT_URL"],
            api_key=DEFAULT_SERVICE_ENV["LIVEKIT_API_KEY"],
            api_secret=DEFAULT_SERVICE_ENV["LIVEKIT_API_SECRET"],
        )
    )
