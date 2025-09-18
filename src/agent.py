import logging
import os
import asyncio

from dotenv import load_dotenv
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
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from .rag_client import retrieve_context  # RAC retrieval via your RAG service

logger = logging.getLogger("agent")

load_dotenv(".env.local")

def validate_environment() -> None:
    """Validate required environment variables at startup."""
    required_vars = {
        "RAG_BASE_URL": "URL for RAG (RAC) service connectivity",
        "LIVEKIT_URL": "LiveKit server URL",
        "LIVEKIT_API_KEY": "LiveKit API key for authentication",
        "LIVEKIT_API_SECRET": "LiveKit API secret for authentication"
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

# Health check function for monitoring RAC connectivity
async def health_check() -> dict:
    """Perform health check for agent and RAC (RAG) connectivity."""
    health_status = {
        "status": "healthy",
        "checks": {
            "environment": {"status": "healthy", "details": {}},
            "rac": {"status": "unknown", "details": {}}
        }
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
        if rag_result and "formattedContext" in rag_result:
            health_status["checks"]["rac"]["status"] = "healthy"
            health_status["checks"]["rac"]["details"]["response"] = "success"
        else:
            health_status["checks"]["rac"]["status"] = "unhealthy"
            health_status["checks"]["rac"]["details"]["response"] = "no_context"
            health_status["status"] = "unhealthy"
    except Exception as e:
        logger.error(f"RAC health check failed: {e}")
        health_status["checks"]["rac"]["status"] = "unhealthy"
        health_status["checks"]["rac"]["details"]["error"] = str(e)
        health_status["status"] = "unhealthy"

    return health_status


# --- BVNM / RAC strict policy injected every turn ---
BVNM_RAC_POLICY = """\
You are BVNM’s voice assistant.
Your ONLY knowledge source is the RAC system provided below as “RAC SNIPPETS”.
Hard rules (must follow):
1) Use ONLY facts that appear explicitly in the RAC SNIPPETS. Do not infer, summarize beyond the text, or reach conclusions.
2) If the RAC SNIPPETS do not contain an answer, say you don’t have that information in RAC and ask a brief, friendly follow-up (e.g., to refine the term).
3) Do NOT use outside knowledge or prior memory.
4) Keep replies short, friendly, and speak naturally. If appropriate, begin with the small filler “M,” once to sound human-like.
5) No emojis or fancy formatting.
If RAC has no relevant snippets, your entire reply should be something like:
“M, I don’t have that in RAC. Can you give me a different term or a bit more detail?”.
"""


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are BVNM’s RAC-only voice assistant. "
                "You must follow the BVNM/RAC policy system messages in this conversation."
            ),
        )

    # Runs AFTER the user finishes speaking and BEFORE the agent speaks.
    # We call our RAC (RAG) HTTP endpoint, then inject strict policy + context.
    async def on_user_turn_completed(self, turn_ctx, new_message):
        try:
            text = (getattr(new_message, "text_content", None) or getattr(new_message, "text", ""))
            if not text:
                return
            if len(text.strip()) < 2:
                return

            # Prefer room name as session id (for tracing)
            session_id = getattr(turn_ctx, "room", None)
            if session_id and hasattr(session_id, "name"):
                session_id = session_id.name
            else:
                session_id = "unknown_session"

            # Retrieve from RAC
            rac = await retrieve_context(question=text, session_id=session_id)
            formatted = (rac or {}).get("formattedContext") or ""

            # Always inject the strict BVNM/RAC policy
            turn_ctx.add_message(role="system", content=BVNM_RAC_POLICY)

            if formatted.strip():
                # Provide snippets for grounded answering
                turn_ctx.add_message(
                    role="system",
                    content=f"RAC SNIPPETS (use only these):\n{formatted}"
                )
            else:
                # No RAC data: instruct the model to politely say it doesn't know (no guessing)
                turn_ctx.add_message(
                    role="system",
                    content=(
                        "RAC SNIPPETS: (none)\n"
                        "There is no relevant RAC content. "
                        "You must answer exactly as instructed in the policy for missing RAC data."
                    ),
                )

        except Exception as e:
            logger.exception(f"pre-reply RAC injection failed: {e}")
            # If RAC fails, still force the assistant to admit it without guessing
            turn_ctx.add_message(
                role="system",
                content=(
                    BVNM_RAC_POLICY
                    + "\nRAC SNIPPETS: (unavailable due to an error)\n"
                      "Answer as if RAC has no data: do not guess."
                ),
            )

    # Keep system-health tool (internal ops); never used for user knowledge
    @function_tool
    async def check_system_health(self, context: RunContext):
        """Check the health status of the agent system including RAC connectivity and environment configuration."""
        try:
            health_result = await health_check()
            status = health_result["status"]

            if status == "healthy":
                summary = "System is healthy. All checks passed."
            else:
                summary = "System has issues that need attention."

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
        llm=openai.LLM(model="gpt-4.1"),
        stt=deepgram.STT(model="nova-3", language="multi"),
        tts=openai.TTS(voice="alloy"),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=False,  # keep pre-reply RAC injection strict
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


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
