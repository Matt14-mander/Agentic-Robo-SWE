"""Explicitly download and verify pinned public ONNX/tokenizer assets; no inference service."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

from agent.rag.embedding import DEFAULT_MODEL_DIR, MODEL_FILES, MODEL_ID, MODEL_REVISION, file_sha256


def download_model(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name, expected in MODEL_FILES.items():
        target = destination / name
        if target.is_file() and file_sha256(target) == expected:
            print(f"Verified cached {name}", flush=True)
            continue
        remote = f"onnx/{name}" if name.endswith(".onnx") else name
        url = f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{remote}"
        temporary = target.with_suffix(target.suffix + ".partial")
        print(f"Downloading {name} from revision {MODEL_REVISION}", flush=True)
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as stream:
            while block := response.read(1024 * 1024):
                stream.write(block)
        if file_sha256(temporary) != expected:
            raise RuntimeError(f"Downloaded checksum mismatch: {name}")
        temporary.replace(target)
        print(f"Verified {name}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    download_model(parser.parse_args().model_dir)
