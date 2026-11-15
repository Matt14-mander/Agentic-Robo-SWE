"""Token coverage, parent context, migration and real local encoder checks."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from agent.rag import chunking, index as index_module
from agent.rag.embedding import (
    DEFAULT_MODEL_DIR, MODEL_FILES, EmbeddingConfig, HashingEmbedder, create_embedder,
)
from agent.rag.index import CodeIndex, open_code_index
from agent.rag.models import CodeChunk
from agent.rag.token_chunking import TokenChunker
from agent.tools import _paths, rag_ops
from tests.test_rag import MemoryCollection
from tests.test_workspace_rag import make_repo


@pytest.fixture
def token_chunker():
    tokenizers = pytest.importorskip("tokenizers")
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel({
        "[UNK]": 0, "[CLS]": 1, "[SEP]": 2,
    }, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Split("", behavior="isolated")
    tokenizer.post_processor = tokenizers.processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
    )
    return TokenChunker(tokenizer, max_tokens=64)


def parent_chunk(text):
    return CodeChunk("parent", "module.py", "controller", "function", 11,
                     11 + text.count("\n") - int(text.endswith("\n")),
                     text, hashlib.sha256(text.encode()).hexdigest())


@pytest.mark.parametrize("text", [
    "x" * 62, "x" * 63, "x" * 4000,
    "@decorator\ndef controller():\n" + "    value += 1\n" * 30,
    "\t中文编码🙂é=1\n" * 70, " " * 200, "", "first\n" + "x" * 1000 + "\nTAIL\n",
])
def test_children_fit_actual_budget_and_cover_every_source_character(token_chunker, text):
    parent = parent_chunk(text)
    children = token_chunker.split(parent)
    assert children == token_chunker.split(parent)
    assert len({child.id for child in children}) == len(children)
    covered = set()
    lines = text.splitlines(keepends=True)
    for child in children:
        assert token_chunker.count(child.content) <= 64
        if child is parent:
            assert len(children) == 1
            covered.update(range(len(text)))
            continue
        assert child.parent_id == parent.id and child.parent_hash == parent.content_hash
        start = sum(map(len, lines[:child.start_line - parent.start_line])) + child.start_column - 1
        end = sum(map(len, lines[:child.end_line - parent.start_line])) + child.end_column
        assert child.content == text[start:end]
        covered.update(range(start, end))
    assert covered == set(range(len(text)))
    if len(text) == 62:
        assert children == [parent]  # Two special tokens fit exactly.
    if len(text) == 63:
        assert len(children) > 1


class BoundedEmbedder(HashingEmbedder):
    def __init__(self, chunker):
        super().__init__()
        self.chunker = chunker

    @property
    def identity(self):
        return "bounded-test-encoder"

    def embed(self, texts):
        assert all(self.chunker.count(text) <= self.chunker.max_tokens for text in texts)
        return [[1.0, 0.0] if "TAIL" in text else [0.0, 1.0] for text in texts]


@pytest.fixture
def token_index(tmp_path, monkeypatch, token_chunker):
    for module in (chunking, index_module, _paths):
        monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    source = tmp_path / "controller.py"
    source.write_text(
        "def controller():\n" + '    value = "' + "x" * 6000 + '"\n    return "TAIL"\n',
        encoding="utf-8",
    )
    index = CodeIndex(MemoryCollection(), embedder=BoundedEmbedder(token_chunker))
    index.sync([source.name])
    return source, index


@pytest.mark.parametrize("strategy", ["lexical", "vector", "hybrid"])
def test_tail_retrieval_keeps_parent_and_tool_prioritizes_matched_child(
    token_index, monkeypatch, strategy,
):
    source, index = token_index
    results = index.search("TAIL", top_k=5, strategy=strategy)
    assert len(results) == 1
    result = results[0]
    assert "TAIL" in result.content
    assert result.parent_context.content == source.read_text(encoding="utf-8")
    assert result.parent_context.start_line == 1 and result.parent_context.end_line == 3
    assert len(result.parent_context.content) > 4000
    monkeypatch.setattr(rag_ops, "get_code_index", lambda: index)
    output = rag_ops.search_code_knowledge.invoke({"query": "TAIL"})
    assert "TAIL" in output and "Parent context" in output
    assert "def controller():" in output and "excerpt" in output
    assert output.index("TAIL") < output.index("Parent context")


def test_many_siblings_do_not_hide_other_parents(token_index):
    source, index = token_index
    source.with_name("other.py").write_text("def other():\n    pass\n", encoding="utf-8")
    index.sync([source.name, "other.py"])
    assert index.collection.count() > 32
    results = index.search("neutral", top_k=2, strategy="vector")
    assert {result.path for result in results} == {"controller.py", "other.py"}
    assert index.collection.query_calls > 1


def test_parent_only_edit_updates_metadata_context_and_cache(token_index):
    source, index = token_index
    first = index.search("TAIL", strategy="vector")[0]
    before = index.cache_stats().revision
    source.write_text(source.read_text().replace("def controller():", "def controller(): # signature"),
                      encoding="utf-8")
    second = index.search("TAIL", strategy="vector")[0]
    assert second.content == first.content
    assert "# signature" in second.parent_context.content
    assert index.cache_stats().revision != before
    assert index.sync([source.name]).added_or_updated == 0


def test_reopen_and_legacy_unsplit_rows_are_rechunked(token_index):
    source, index = token_index
    raw = chunking.chunk_python_file(source)[0]
    legacy = MemoryCollection()
    legacy.upsert(ids=[raw.id], documents=[raw.content], metadatas=[raw.metadata()],
                  embeddings=[[0.0, 1.0]])
    reopened = CodeIndex(legacy, embedder=index.embedder)
    result = reopened.search("TAIL", strategy="vector")[0]
    assert "TAIL" in result.content and result.parent_context.content == raw.content
    assert raw.id not in legacy.data
    assert legacy.count() == index.collection.count()
    assert CodeIndex(legacy, embedder=index.embedder).search("TAIL")[0].parent_context


def test_source_revision_is_independent_of_token_chunk_layout(token_index):
    source, semantic = token_index
    hashed = CodeIndex(MemoryCollection())
    hashed.sync([source.name])
    assert hashed.retrieval_manifest()["source_revision"] == semantic.retrieval_manifest()["source_revision"]
    assert hashed.retrieval_manifest()["index_revision"] != semantic.retrieval_manifest()["index_revision"]
    assert semantic.retrieval_manifest()["parent_count"] == 1


def test_parent_change_invalidates_identical_child_fingerprint(token_chunker):
    first = parent_chunk("x" * 200 + "TAIL")
    second = replace(first, content="x" * 200 + "EDIT", content_hash="new-parent-hash")
    one, two = token_chunker.split(first)[0], token_chunker.split(second)[0]
    assert one.id == two.id and one.content_hash == two.content_hash
    assert one.revision_hash != two.revision_hash


@pytest.mark.parametrize("max_tokens", [8, 32, 128])
def test_smaller_budgets_include_special_tokens_and_make_progress(token_chunker, max_tokens):
    chunker = TokenChunker(token_chunker.tokenizer, max_tokens=max_tokens)
    children = chunker.split(parent_chunk("中文🙂value=" * 80))
    assert all(chunker.count(child.content) <= max_tokens for child in children)
    assert len(children) <= 800


def test_external_cpp_children_refresh_and_preserve_parent_scope(tmp_path, token_chunker):
    workspace = make_repo(tmp_path)
    source = workspace.root / "src/controller.cpp"
    source.write_text("#define LIMIT 42\n" + "// padding\n" * 70 + "// TAIL\n", encoding="utf-8")
    index = CodeIndex(MemoryCollection(), embedder=BoundedEmbedder(token_chunker), workspace=workspace)
    index.sync(["src"])
    first = index.search("TAIL", path_prefix="src", strategy="vector")[0]
    assert first.parent_context.content == source.read_text()
    assert first.parent_context.start_line == 1 and first.parent_context.end_line == 72
    source.write_text("// TAIL\n", encoding="utf-8")
    second = index.search("TAIL", strategy="vector")[0]
    assert second.content == "// TAIL\n" and second.parent_context is None
    assert index.collection.count() == 1
    source.unlink()
    assert index.search("TAIL") == []


def test_real_chroma_roundtrip_keeps_parent_context(token_index, tmp_path, monkeypatch):
    pytest.importorskip("chromadb")
    monkeypatch.delenv("SSLKEYLOGFILE", raising=False)
    source, index = token_index
    monkeypatch.setattr(index_module, "create_embedder", lambda config: index.embedder)
    persistent = open_code_index(tmp_path / "db")
    persistent.sync([source.name])
    reopened = open_code_index(tmp_path / "db")
    assert reopened.search("TAIL", strategy="vector")[0].parent_context.content == source.read_text()


def test_real_semantic_children_encode_without_truncation(tmp_path):
    if not all((DEFAULT_MODEL_DIR / name).is_file() for name in MODEL_FILES):
        pytest.skip("Local code model assets not installed")
    pytest.importorskip("onnxruntime")
    model = create_embedder(EmbeddingConfig(backend="semantic"))
    text = "double controller() {\n" + (
        "    // static_cast<double>(controller.compute(position, velocity, dt));\n" * 100
    ) + "    return emergency_shutdown_after_sensor_fault;\n}\n"
    parent = parent_chunk(text)
    model.embed([text])
    assert model.truncated_texts == 1  # Reproduce the old document embedding path.
    children = model.chunker.split(parent)
    assert len(children) > 1 and any("emergency_shutdown" in c.content for c in children)
    vectors = model.embed([child.content for child in children])
    assert len(vectors) == len(children)
    assert all(model.chunker.count(child.content) <= 512 for child in children)
    assert model.truncated_texts == 1  # No child adds an overflow.
