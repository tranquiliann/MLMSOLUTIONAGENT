import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("RAG_BASE_URL", "http://localhost:8000")
os.environ.setdefault("LIVEKIT_URL", "wss://dummy.livekit.local")
os.environ.setdefault("LIVEKIT_API_KEY", "test-key")
os.environ.setdefault("LIVEKIT_API_SECRET", "test-secret")
os.environ.setdefault("OPENAI_API_KEY", "sk-test")
os.environ.setdefault("SKIP_TURN_DETECTOR_CHECK", "1")
os.environ.setdefault("DISABLE_RERANKER", "1")
