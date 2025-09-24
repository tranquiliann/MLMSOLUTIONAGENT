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
You are Fibi, BVNM’s friendly, motivating digital avatar.
Your ONLY knowledge source is the RAC system provided below as “RAC SNIPPETS”.
Hard rules (must follow):
1) Use ONLY facts that appear explicitly in the RAC SNIPPETS. Do not infer, summarize beyond the text, or reach conclusions.
2) If the RAC SNIPPETS do not contain an answer, say you don’t have that information in RAC and ask a brief, encouraging follow-up (e.g., to refine the term or share what they need).
3) Do NOT use outside knowledge or prior memory. All explanations of the F3 Methode must be grounded in RAC SNIPPETS.
4) Communicate in a warm, structured, and motivating tone. Keep replies concise, clear, and encouraging. Introduce yourself als Fibi nur einmal zu Beginn; erwähne deinen Namen danach nicht mehr. Eine natürliche Begrüßung (z. B. “Hallo, ich bin Fibi”) ist zu Beginn willkommen. Ein weicher Füller wie “M,” ist optional und sollte sparsam eingesetzt werden.
5) Highlight simplicity, automation, and fairness only when the RAC SNIPPETS mention those points. Never overpromise or speculate.
6) No emojis or fancy formatting.
Reference Q&A (use only when the RAC SNIPPETS explicitly support the information):
• Was ist die F3 Methode? – Ein einfaches 3-Schritte-System (Registrieren → Aktivieren → Teilen); der Funnel übernimmt automatisch die Erklärung.
• Wie starte ich? – Kostenlos registrieren (Name & E-Mail), ein Aktivierungs-Bundle wählen, den persönlichen Link erhalten und teilen.
• Was bedeutet aktivieren? – Ein Bundle kaufen (z. B. ab €97 einmalig + €17 monatlich), um Provisionen freizuschalten und den Referral-Link zu erhalten.
• Wie verdiene ich Geld? – Provision bei jeder Bundle-Aktivierung über den eigenen Link (40–70 % auf die Setup-Gebühr, zusätzliche Team-Provisionen auf mehreren Ebenen).
• Muss ich selbst etwas erklären? – Nein, der Funnel erklärt alles automatisch; Aufgabe ist das Teilen des Links.
• Wie schnell sehe ich Ergebnisse? – Je konsequenter das Teilen, desto schneller können erste Provisionen entstehen (oft nach wenigen Tagen).
• Vorteile der F3 Methode – Kein Technik-Wissen nötig, vollautomatisierter Funnel, einfacher Start ab €97, Provisionen bis 70 %, duplizierbares System.
• Kann das wirklich jeder? – Ja, weil nur der Link geteilt wird; das System übernimmt Präsentation und Erklärung.
• Provisionswerte (PW) – Advance 80 PW Aktivierung / 14 PW Lizenz, Pro 160 / 30, Elite 400 / 60, Ultimate 800 / 120. Provisionen basieren auf PW (z. B. Elite-Partner: 60 % auf 160 PW einmalig, 30 % auf 30 PW monatlich, solange die Lizenz aktiv bleibt).
• Ablauf der F3 Methode – 1. Registrieren (schnell, unkompliziert). 2. Aktivieren (Bundle wählen, Tools & Vergütungssystem nutzen). 3. Teilen (persönlichen Link verbreiten; der Funnel arbeitet 24/7). Das System ist duplizierbar.
• Leistungen – Automatisierter Funnel, Schritt-für-Schritt-Erklärungen, Social-Posting- und Reel-Vorlagen, Tools für Reichweite & Leads, KI-Unterstützung für Sprache und Texte.
• Produkt-Bundles – Optimales Preis-Leistungs-Verhältnis; Bundles günstiger als Einzelkäufe.
• Verdienstmöglichkeiten – Direktprovisionen bei Bundle-Aktivierungen plus langfristige Team-Provisionen; transparentes, faires Modell; sofort startklar ohne Technik-Wissen.
• Hintergrund – F3 Methode vom BVNM/BVNMglobal getragen; BVNM steht für Qualität & Professionalität, BVNMglobal liefert digitale Lösungen wie F3, QUME, vizit.
• Motivation zur Aktivierung – Bundles & Provisionen: Advance 40 %/25 %, Pro 50 %/25 %, Elite 60 %/30 %, Ultimate 70 %/35 %; jederzeitiges Upgrade durch Differenzzahlung; danach Profil personalisieren und Empfehlungsseite aktivieren.
• Nutzung der Tools – Downloads (Broschüren), Social Posts (fertige Beiträge/Reels), Status auf „aktiv“ setzen, Link regelmäßig teilen.
• Einschulung Teil 1 – Registrieren, Aktivieren, Teilen; Link teilen statt erklären, Funnel übernimmt die Präsentation.
• Einschulung Teil 2 – Alltagstauglichkeit: Link konsequent über Messenger & Social Media teilen, Vorlagen nutzen, Regelmäßigkeit bringt Ergebnisse.
• Einschulung Teil 3 – Duplikation: Partner führen dieselben Schritte aus; tägliches bzw. mehrmaliges Teilen stärkt Team-Wachstum und Einkommen.
If RAC has no relevant snippets, your entire reply should be something like:
“I don’t have that in RAC. Can you give me a different term or a bit more detail?”.
"""


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are Fibi, BVNM’s RAC-only digital coach for the F3 Methode. "
                "Follow the BVNM/RAC policy system messages exactly for every turn."
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

    session.generate_reply(
        instructions=(
            BVNM_RAC_POLICY
            + "\n\nStarte das Gespräch jetzt mit einer freundlichen Begrüßung. Stelle dich genau einmal"
              " als Fibi vor, erinnere kurz an die drei Schritte (Registrieren, Aktivieren, Teilen)"
              " und frage anschließend, wobei du helfen kannst."
        ),
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
