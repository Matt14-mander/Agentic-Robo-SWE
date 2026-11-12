"""Source coverage and retrieval regressions for structure-aware Python chunks."""

from __future__ import annotations

import hashlib
from textwrap import dedent

import pytest

from agent.rag.chunking import chunk_python_file
from agent.rag.index import CodeIndex
from tests.test_rag import MemoryCollection


@pytest.fixture
def source_file(tmp_path, monkeypatch):
    from agent.rag import chunking, index
    from agent.tools import _paths

    for module in (chunking, index, _paths):
        monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    return tmp_path / "example.py"


def assert_source_covered(source, chunks):
    lines = source.splitlines()
    covered = set()
    assert len({chunk.id for chunk in chunks}) == len(chunks)
    for chunk in chunks:
        expected = "\n".join(lines[chunk.start_line - 1:chunk.end_line]).rstrip() + "\n"
        assert chunk.content == expected
        assert chunk.content_hash == hashlib.sha256(expected.encode("utf-8")).hexdigest()
        covered.update(range(chunk.start_line, chunk.end_line + 1))
    assert {i for i, line in enumerate(lines, 1) if line.strip()} <= covered


def test_module_statements_and_comments_between_and_after_definitions(source_file):
    source = dedent('''\
        import math
        def first():
            return 1

        # Runtime configuration must remain searchable.
        LIMIT: int = 42
        from collections import deque
        if LIMIT:
            queue = deque()

        def second():
            return LIMIT

        REGISTRY = {"second": second}
        if __name__ == "__main__":
            second()
        # End-of-file operational note.
    ''')
    source_file.write_text(source, encoding="utf-8")
    chunks = chunk_python_file(source_file)
    assert_source_covered(source, chunks)
    for token in ("LIMIT: int", "REGISTRY =", "End-of-file", "if __name__"):
        assert any(token in chunk.content and chunk.kind == "module" for chunk in chunks)


@pytest.mark.parametrize("decorator", [
    "@registered",
    "@registered(\n    mode='safe',\n)",
    "@(\n    registered\n)",
    "@first\n@second",
    "@(\n    left\n    @ right\n)",
])
def test_decorators_belong_to_their_definitions(source_file, decorator):
    source = (
        f"{decorator}\nasync def first():\n    return 1\n\n"
        f"{decorator}\nclass Controller:\n"
        "    @staticmethod\n    def first():\n        return 1\n\n"
        "    @registered(\n        mode='safe',\n    )\n"
        "    async def second(self):\n        return 2\n"
    )
    source_file.write_text(source, encoding="utf-8")
    chunks = chunk_python_file(source_file)
    assert_source_covered(source, chunks)
    by_symbol = {chunk.symbol: chunk for chunk in chunks}
    assert by_symbol["first"].content.startswith(decorator)
    assert by_symbol["Controller"].content.startswith(decorator)
    assert by_symbol["Controller.first"].content.lstrip().startswith("@staticmethod")
    assert by_symbol["Controller.second"].content.lstrip().startswith("@registered(")
    for number, line in enumerate(source.splitlines(), 1):
        if line.lstrip().startswith("@"):
            owners = [chunk for chunk in chunks if chunk.start_line <= number <= chunk.end_line]
            assert len(owners) == 1
            assert owners[0].start_line <= number
            assert owners[0].kind != "module"


def test_class_body_after_methods_and_conditional_definitions_are_preserved(source_file):
    source = dedent('''\
        class Controller:
            def first(self):
                return 1
            # These statements are not part of the preceding method.
            LIMIT = 42
            class Nested:
                def inner(self):
                    return 3
            @property
            def second(self):
                return self.LIMIT
            ALIAS = second

        if True:
            @registered
            def conditional():
                return 2
    ''')
    source_file.write_text(source, encoding="utf-8")
    chunks = chunk_python_file(source_file)
    assert_source_covered(source, chunks)
    for token in ("LIMIT =", "class Nested", "ALIAS ="):
        assert any(token in chunk.content and chunk.kind == "class" for chunk in chunks)
    assert any("def conditional" in chunk.content and chunk.kind == "module" for chunk in chunks)


def test_long_residual_regions_keep_chunk_limits_and_full_coverage(source_file):
    source = "def first():\n    pass\n" + "".join(f"VALUE_{i} = {i}\n" for i in range(25))
    source += "@registered\ndef long_function():\n"
    source += "".join(f"    value_{i} = {i}\n" for i in range(25))
    source += "class Controller:\n    def first(self):\n        pass\n"
    source += "".join(f"    VALUE_{i} = {i}\n" for i in range(25))
    source_file.write_text(source, encoding="utf-8")
    chunks = chunk_python_file(source_file, max_lines=10, overlap=3)
    assert_source_covered(source, chunks)
    assert all(chunk.end_line - chunk.start_line + 1 <= 10 for chunk in chunks)
    assert any(chunk.symbol.startswith("<module>#part") for chunk in chunks)
    assert any(chunk.symbol.startswith("Controller#part") for chunk in chunks)


def test_syntax_error_fallback_and_source_snapshot_preserve_content(source_file):
    source_file.write_text("# unrelated disk contents\n", encoding="utf-8")
    source = "@registered\ndef incomplete(\nCONFIG = 42\n"
    chunks = chunk_python_file(source_file, source_text=source)
    assert_source_covered(source, chunks)
    assert all(chunk.kind == "module" for chunk in chunks)


def test_missing_regions_are_recovered_in_old_index_and_refreshed_after_patch(source_file):
    from agent.tools.write_ops import write_patch

    source = 'def first():\n    pass\nCONFIG = "unique_config_before"\n@special_wrapper\ndef second():\n    pass\n'
    source_file.write_text(source, encoding="utf-8")
    collection = MemoryCollection()
    index = CodeIndex(collection)
    index.sync(["example.py"])
    # Emulate an old index containing only the two undecorated functions.
    legacy = chunk_python_file(source_file, source_text=source.replace(
        'CONFIG = "unique_config_before"', ""
    ).replace("@special_wrapper", ""))
    collection.data.clear()
    for chunk in legacy:
        collection.upsert([chunk.id], [chunk.content], [chunk.metadata()],
                          index.embedder.embed([chunk.content]))
    reopened = CodeIndex(collection)
    result = reopened.search("unique_config_before", strategy="lexical")[0]
    assert result.kind == "module"
    decorated = reopened.search("special_wrapper", strategy="lexical")[0]
    assert decorated.symbol == "second" and decorated.start_line == 4
    status = write_patch.invoke({"path": "example.py", "old_string": "unique_config_before",
                                 "new_string": "unique_config_after"})
    assert status.startswith("OK:")
    assert "unique_config_after" in reopened.search("unique_config_after", strategy="lexical")[0].content
    assert index.sync(["example.py"]).added_or_updated == 0
