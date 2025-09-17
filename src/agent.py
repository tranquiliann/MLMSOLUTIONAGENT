import logging
import os

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

from .rag_client import retrieve_context  # <-- NEW: async HTTP call to your RAG infra

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice AI assistant.
            You eagerly assist users with their questions by providing information from your extensive knowledge.
            Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
            You are curious, friendly, and have a sense of humor.""",
        )

    # --- LIVEKIT DOCS RAG: pre-reply hook pattern (adapted) ---
    # Runs AFTER the user finishes speaking and BEFORE the agent speaks.
    # We call our RAG HTTP endpoint, then inject context into the turn.
    async def on_user_turn_completed(self, turn_ctx, new_message):
        try:
            text = new_message.text_content()
            if not text:
                return

            # Optional: short-circuit non-question noise
            if len(text.strip()) < 2:
                return

            # Read a session ID for tracing; prefer room name
            session_id = getattr(turn_ctx, "room", None)
            if session_id and hasattr(session_id, "name"):
                session_id = session_id.name
            else:
                session_id = "unknown_session"

            # Call your Next/Redis RAG service
            rag = await retrieve_context(question=text, session_id=session_id)

            formatted = (rag or {}).get("formattedContext") or ""
            if not formatted.strip():
                # Nothing to inject—let the LLM proceed with its base instructions
                return

            # Inject a system message (context) BEFORE generation
            # This mirrors the Docs’ pre-reply injection timing.
            # The framework will use this updated context for the imminent reply.
            turn_ctx.add_message(role="system", content=formatted)

            # NOTE: Most recent LiveKit Agents will consume turn_ctx additions automatically
            # for the upcoming generation. If your version exposes a 'commit' or 'update_chat_ctx'
            # helper, you could call it here. Otherwise, adding the message is sufficient.

        except Exception as e:
            logger.exception(f"pre-reply RAG hook failed: {e}")

    # Demo tool remains (unchanged)
    @function_tool
    async def lookup_weather(self, context: RunContext, location: str):
        """Use this tool to look up current weather information in the given location.
        If the location is not supported by the weather service, the tool will indicate this. You must tell the user the location's weather is unavailable.
        Args:
            location: The location to look up weather information for (e.g. city name)
        """
        logger.info(f"Looking up weather for {location}")
        return "sunny with a temperature of 70 degrees."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # --- IMPORTANT: disable preemptive_generation for true pre-reply RAG ---
    session = AgentSession(
        llm=openai.LLM(model="gpt-4o-mini"),
        stt=openai.STT(model="gpt-4o-transcribe"),
        tts=openai.TTS(voice="alloy"),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=False,  # <-- changed from True
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
