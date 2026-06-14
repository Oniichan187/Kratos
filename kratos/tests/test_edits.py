"""Tests for the surgical search/replace edit primitive (execution/edits.py)."""

from kratos.execution.edits import parse_edit_blocks, apply_search_replace


def test_exact_unique_replace():
    new, status = apply_search_replace("a\nb\nc\n", "b", "B")
    assert status == "ok"
    assert new == "a\nB\nc\n"


def test_not_found_leaves_content_unchanged():
    new, status = apply_search_replace("a\nb\n", "zzz", "q")
    assert status == "not_found"
    assert new == "a\nb\n"


def test_ambiguous_replaces_first_only():
    new, status = apply_search_replace("x\nx\n", "x", "y")
    assert status == "ambiguous"
    assert new == "y\nx\n"


def test_empty_search_is_rejected():
    new, status = apply_search_replace("a\n", "   ", "b")
    assert status == "empty_search"
    assert new == "a\n"


def test_noop_when_search_equals_replace():
    new, status = apply_search_replace("a\n", "a", "a")
    assert status == "noop"


def test_normalized_match_tolerates_trailing_whitespace_keeps_indent():
    content = "def f():\n    return 1   \nx = 0\n"
    new, status = apply_search_replace(content, "    return 1\nx = 0", "    return 2\nx = 1")
    assert status == "ok_normalized"
    assert "    return 2" in new
    assert "return 1" not in new


def test_exact_match_preserves_crlf():
    content = "line1\r\nold\r\nline3\r\n"
    new, status = apply_search_replace(content, "old", "new")
    assert status == "ok"
    assert new == "line1\r\nnew\r\nline3\r\n"


def test_multiline_search_replace():
    content = "head\nint x = 1\nint y = 2\ntail\n"
    new, status = apply_search_replace(content, "int x = 1\nint y = 2", "int x = 10")
    assert status == "ok"
    assert new == "head\nint x = 10\ntail\n"


def test_parse_single_block():
    text = (
        "### EDIT: pkg/mod.py\n"
        "<<<<<<< SEARCH\n"
        "old line\n"
        "=======\n"
        "new line\n"
        ">>>>>>> REPLACE\n"
    )
    assert parse_edit_blocks(text) == [("pkg/mod.py", "old line", "new line")]


def test_parse_multiple_blocks():
    text = (
        "### EDIT: a.py\n<<<<<<< SEARCH\nfoo\n=======\nbar\n>>>>>>> REPLACE\n"
        "some prose\n"
        "### EDIT: b.py\n<<<<<<< SEARCH\nx\n=======\ny\n>>>>>>> REPLACE\n"
    )
    blocks = parse_edit_blocks(text)
    assert len(blocks) == 2
    assert blocks[0][0] == "a.py" and blocks[1][0] == "b.py"


def test_parse_tolerates_backticked_path():
    text = "### EDIT: `pkg/x.py`\n<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE\n"
    assert parse_edit_blocks(text)[0][0] == "pkg/x.py"


def test_parse_ignores_non_edit_text():
    assert parse_edit_blocks("just text\n### FILE: x.py\n```\nhi\n```\n") == []
