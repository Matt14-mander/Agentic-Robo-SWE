"""Shared explicit embedding options for indexing and evaluation."""

from argparse import ArgumentParser, Namespace
from dataclasses import replace
from pathlib import Path

from agent.rag.embedding import EmbeddingConfig


def add_embedding_arguments(parser: ArgumentParser) -> None:
    parser.add_argument("--embedding", choices=("auto", "hash", "semantic"),
                        help="auto uses verified local code model, otherwise reports hash fallback")
    parser.add_argument("--embedding-model-dir", type=Path, help="Local pinned ONNX/tokenizer directory")
    parser.add_argument("--embedding-max-tokens", type=int, help="Encoder token limit (default 512)")


def embedding_config(args: Namespace) -> EmbeddingConfig:
    config = EmbeddingConfig.from_env()
    return replace(config, backend=args.embedding or config.backend,
                   model_dir=(args.embedding_model_dir.resolve() if args.embedding_model_dir
                              else config.model_dir),
                   max_tokens=(args.embedding_max_tokens if args.embedding_max_tokens is not None
                               else config.max_tokens))
