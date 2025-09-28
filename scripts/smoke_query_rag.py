"""Manual smoke-test for the unified RAG query flow."""

import asyncio
from types import SimpleNamespace

from agent import Assistant


async def main() -> None:
    ctx = SimpleNamespace(
        session=SimpleNamespace(
            room=SimpleNamespace(name="smoke-room"),
        )
    )

    assistant = Assistant()
    answer = await assistant.query_rag(ctx, "Wie funktioniert die F3-Methode?")
    print("RAG answer:\n", answer)


if __name__ == "__main__":
    asyncio.run(main())
