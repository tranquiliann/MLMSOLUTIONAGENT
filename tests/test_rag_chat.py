from types import SimpleNamespace

import pytest

import rag_chat
from rag_chat import RemoteRAGRetriever
from llama_index.core.schema import QueryBundle


@pytest.mark.asyncio
async def test_remote_rag_retriever_aretrieve_maps_items(monkeypatch):
    async def fake_rag_search(question, *, session_id=None, top_k=None, base_url=None):
        return {
            "items": [
                {
                    "text": "Antwort aus der Wissensbasis.",
                    "score": 0.42,
                    "id": "doc:1",
                    "source": "kb",
                    "metadata": {
                        "distance": 0.42,
                        "source": "kb",
                        "title": "BVNM FAQ",
                    },
                }
            ]
        }

    monkeypatch.setattr(rag_chat, "rag_search", fake_rag_search)

    retriever = RemoteRAGRetriever(top_k=3, base_url="http://rag")
    retriever.set_session("session-1")
    nodes = await retriever.aretrieve(QueryBundle(query_str="Was ist BVNM?"))

    assert len(nodes) == 1
    node_with_score = nodes[0]
    assert node_with_score.node.text == "Antwort aus der Wissensbasis."
    assert node_with_score.node.metadata["distance"] == pytest.approx(0.42)
    assert node_with_score.score == pytest.approx(-0.42)
    assert node_with_score.node.metadata["source"] == "kb"


@pytest.mark.asyncio
async def test_query_rag_returns_answer_for_single_hit(monkeypatch):
    async def fake_rag_search(question, *, session_id=None, top_k=None, base_url=None):
        return {
            "items": [
                {
                    "text": "BVNM steht für Bundesverband Network Marketing.",
                    "score": 0.05,
                    "id": "doc:42",
                    "metadata": {"distance": 0.05, "source": "kb"},
                }
            ],
            "formattedContext": "[1] BVNM steht für Bundesverband Network Marketing.",
        }

    class DummyLLM:
        def __init__(self, *_, **__):
            pass

    class DummySynthesizer:
        callback_manager = None

    class DummyChatEngine:
        def __init__(self, query_engine):
            self._query_engine = query_engine

        async def achat(self, question):
            nodes = await self._query_engine._retriever.aretrieve(QueryBundle(query_str=question))
            joined = " ".join(node.node.text for node in nodes) or "keine daten"
            return SimpleNamespace(response=joined)

        def reset(self):
            pass

    def fake_from_defaults(cls, *, query_engine, **_):
        return DummyChatEngine(query_engine)

    def fake_get_response_synthesizer(**_):
        return DummySynthesizer()

    def fake_memory_from_defaults(cls, **_):
        return object()

    monkeypatch.setattr(rag_chat, "rag_search", fake_rag_search)
    monkeypatch.setattr(rag_chat, "LlamaOpenAI", DummyLLM)
    monkeypatch.setattr(rag_chat, "get_response_synthesizer", fake_get_response_synthesizer)
    monkeypatch.setattr(rag_chat.ChatMemoryBuffer, "from_defaults", classmethod(fake_memory_from_defaults))
    monkeypatch.setattr(
        rag_chat.CondenseQuestionChatEngine,
        "from_defaults",
        classmethod(fake_from_defaults),
    )

    from agent import Assistant

    assistant = Assistant()
    ctx = SimpleNamespace(
        session=SimpleNamespace(room=SimpleNamespace(name="room-42"))
    )

    answer = await assistant.query_rag(ctx, "Was bedeutet BVNM?")

    assert answer
    assert "Bundesverband" in answer
