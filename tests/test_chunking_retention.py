"""Nonempty structure sections survive chunking, including short sections."""

import pytest

from src.chunking import StructureAwareChunker
from src.file_service import DocumentProcessor


@pytest.mark.parametrize("text", [
    "# Checklist\n\nquasar launch",
    "quasar preface\n\n# Deployment\n" + "zenith deployment guide " * 12 + "\n## Reminder\nmariner rollback",
])
def test_markdown_keeps_short_sections(text):
    chunks = StructureAwareChunker().chunk_markdown(text)
    content = "\n".join(chunk.content for chunk in chunks)
    assert "quasar" in content
    if "mariner" in text:
        assert "mariner" in content
        assert "zenith" in content


def test_html_keeps_short_sections():
    chunks = StructureAwareChunker().chunk_html("<h1>Checklist</h1><p>quasar launch</p><h2>Reminder</h2><p>mariner rollback</p>")
    content = "\n".join(chunk.content for chunk in chunks)
    assert "quasar launch" in content
    assert "mariner rollback" in content


@pytest.mark.parametrize("suffix,content", [
    (".md", "# Checklist\n\nquasar launch\n\n## Reminder\nmariner rollback"),
    (".html", "<h1>Checklist</h1><p>quasar launch</p><h2>Reminder</h2><p>mariner rollback</p>"),
])
def test_file_processor_preserves_structure_content(tmp_path, settings_env, monkeypatch, suffix, content):
    settings_env(DOCUMENTS_PATH=tmp_path, CODE_PATH=tmp_path / "code")
    # HTML is opt-in; do not broaden the production file allowlist.
    from src.config import get_file_whitelist
    whitelist = get_file_whitelist()
    monkeypatch.setattr(whitelist, "_allowed_extensions", [*whitelist.allowed_extensions, ".html"])
    path = tmp_path / ("checklist" + suffix)
    path.write_text(content)
    chunks = list(DocumentProcessor().process_file(path))
    result = "\n".join(chunk.content for chunk in chunks)
    assert "quasar launch" in result
    assert "mariner rollback" in result
    assert all(chunk.metadata["source"] == str(path) for chunk in chunks)


@pytest.mark.parametrize("text,expected", [
    ("<h1>quasar launch checklist</h1>", ["quasar launch checklist"]),
    ("<body>quasar launch checklist</body>", ["quasar launch checklist"]),
    ("<p>quasar launch</p><table><tr><td>mariner rollback</td></tr></table>", ["quasar launch", "mariner rollback"]),
    ("<div><p>quasar <strong>launch</strong></p></div>", ["quasar", "launch"]),
])
def test_html_preserves_visible_text_once(text, expected):
    content = "\n".join(chunk.content for chunk in StructureAwareChunker().chunk_html(text))
    for fragment in expected:
        assert content.count(fragment) == 1


def test_html_excludes_non_content_text():
    text = "<html><head><title>hidden title</title><style>hidden style</style></head><body><!-- hidden comment --><script>hidden script</script><p>quasar launch</p></body></html>"
    content = "\n".join(chunk.content for chunk in StructureAwareChunker().chunk_html(text))
    assert "quasar launch" in content
    assert "hidden" not in content


@pytest.mark.parametrize("markup,expected", [
    ("<p>qu<span>asar</span> launch</p>", "quasar launch"),
    ("<p>quasar <strong>launch</strong> checklist</p>", "quasar launch checklist"),
    ("<p>quasar</p><p>launch</p>", "quasar\nlaunch"),
    ("<p>quasar<br>launch</p>", "quasar\nlaunch"),
])
def test_html_preserves_inline_adjacency_and_block_boundaries(markup, expected):
    content = "\n".join(chunk.content for chunk in StructureAwareChunker().chunk_html(markup))
    assert content == expected
