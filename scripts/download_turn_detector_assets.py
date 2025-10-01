import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from livekit.plugins.turn_detector.models import (
    HG_MODEL,
    MODEL_REVISIONS,
    ONNX_FILENAME,
)

REPO_ID = HG_MODEL
REVISION = MODEL_REVISIONS["multilingual"]

# filename, kwargs passed to hf_hub_download (e.g. subfolder)
HF_FILES = [
    ("languages.json", {}),
    ("tokenizer.json", {}),
    ("tokenizer_config.json", {}),
    ("special_tokens_map.json", {}),
    ("vocab.json", {}),
    ("merges.txt", {}),
    ("config.json", {}),
    ("generation_config.json", {}),
    ("added_tokens.json", {}),
    (ONNX_FILENAME, {"subfolder": "onnx"}),
]


def allow_patterns() -> list[str]:
    patterns: set[str] = set()
    for filename, kwargs in HF_FILES:
        subfolder = kwargs.get("subfolder")
        if subfolder:
            patterns.add(f"{subfolder}/{filename}")
        else:
            patterns.add(filename)
    return sorted(patterns)


def main() -> None:
    cache_dir = Path(os.environ.get("HUGGINGFACE_HUB_CACHE") or os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    cache_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=REPO_ID,
        revision=REVISION,
        cache_dir=str(cache_dir),
        resume_download=True,
        allow_patterns=allow_patterns(),
        local_files_only=False,
    )

    missing_files: list[str] = []
    for filename, kwargs in HF_FILES:
        try:
            hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                revision=REVISION,
                cache_dir=str(cache_dir),
                local_files_only=True,
                **kwargs,
            )
        except Exception:
            if kwargs.get("subfolder"):
                missing_files.append(f"{kwargs['subfolder']}/{filename}")
            else:
                missing_files.append(filename)

    if missing_files:
        for path in missing_files:
            print(f"Missing required HF asset: {path}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
