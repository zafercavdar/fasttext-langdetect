"""Integration tests for :mod:`ftlangdetect.detect`.

These tests download the fastText model on first run (cached under
``$FTLANG_CACHE`` or the system temp dir).
"""

from __future__ import annotations

import importlib

import pytest

from ftlangdetect import DetectionResult, detect

detect_module = importlib.import_module("ftlangdetect.detect")


@pytest.mark.parametrize(
    ("text", "expected_lang"),
    [
        ("Bugün hava çok güzel", "tr"),
        ("The quick brown fox jumps over the lazy dog", "en"),
        ("Le chat dort sur le canapé", "fr"),
        ("Das Wetter ist heute schön", "de"),
        ("今日はとても良い天気です", "ja"),
    ],
)
@pytest.mark.parametrize("low_memory", [True, False])
def test_detect_returns_expected_language(text: str, expected_lang: str, low_memory: bool) -> None:
    result = detect(text, low_memory=low_memory)

    assert isinstance(result, dict)
    assert set(result) == {"lang", "score"}
    assert isinstance(result["lang"], str)
    assert isinstance(result["score"], float)
    assert 0.0 <= result["score"] <= 1.0
    assert result["lang"] == expected_lang


def test_detect_result_is_typed_dict_compatible() -> None:
    result: DetectionResult = detect("hello world", low_memory=True)
    assert result["lang"] == "en"


@pytest.mark.parametrize(
    "text",
    [
        "line one\nline two",
        "line one\r\nline two",
        "first paragraph\n\nsecond paragraph",
        "col1\tcol2\tcol3",
        "  leading and trailing spaces  ",
        "many\u00a0\u00a0non\u00a0breaking\u00a0spaces",
        "mixed\n\twhitespace \r\n everywhere",
    ],
)
def test_detect_handles_arbitrary_whitespace(text: str) -> None:
    """detect() normalizes any whitespace (newlines, tabs, NBSP, ...) to spaces."""
    result = detect(text, low_memory=True)
    assert isinstance(result, dict)
    assert isinstance(result["lang"], str)
    assert 0.0 <= result["score"] <= 1.0


def test_detect_multiline_matches_single_line() -> None:
    """The same content split across lines should predict the same language."""
    single = detect("The quick brown fox jumps over the lazy dog", low_memory=False)
    multi = detect(
        "The quick brown fox\njumps over the lazy dog",
        low_memory=False,
    )
    assert single["lang"] == multi["lang"] == "en"


@pytest.mark.parametrize("text", ["", "   ", "\n\n", "\t \r\n  "])
def test_detect_raises_on_empty_or_whitespace_only(text: str) -> None:
    with pytest.raises(ValueError, match=r"empty|whitespace"):
        detect(text, low_memory=True)


def test_detect_explicit_k_equals_1_returns_single_dict() -> None:
    """Backward compat: explicit k=1 still returns a single dict, not a list."""
    result = detect("hello world", low_memory=True, k=1)
    assert isinstance(result, dict)
    assert result["lang"] == "en"


@pytest.mark.parametrize("k", [2, 3, 5])
@pytest.mark.parametrize("low_memory", [True, False])
def test_detect_returns_top_k_sorted(k: int, low_memory: bool) -> None:
    """When k>1 the API returns a list of DetectionResults sorted by score."""
    text = "Hello world. Bonjour le monde. Hola mundo."
    results = detect(text, low_memory=low_memory, k=k)

    assert isinstance(results, list)
    assert 1 <= len(results) <= k
    for item in results:
        assert isinstance(item, dict)
        assert set(item) == {"lang", "score"}
        assert isinstance(item["lang"], str)
        assert isinstance(item["score"], float)
        assert 0.0 <= item["score"] <= 1.0

    scores = [item["score"] for item in results]
    assert scores == sorted(scores, reverse=True), "results must be sorted by descending score"

    langs = [item["lang"] for item in results]
    assert len(set(langs)) == len(langs), "languages in top-k should be distinct"


def test_detect_k_zero_raises() -> None:
    with pytest.raises(ValueError, match="k must be >= 1"):
        detect("hello world", low_memory=True, k=0)


def test_detect_k_negative_raises() -> None:
    with pytest.raises(ValueError, match="k must be >= 1"):
        detect("hello world", low_memory=True, k=-3)


def test_detect_multilingual_top_two_contains_expected_languages() -> None:
    """Top-2 detection on a clearly bilingual sentence should surface both languages."""
    text = "The quick brown fox. Le chat dort sur le canapé."
    results = detect(text, low_memory=False, k=2)

    assert isinstance(results, list)
    assert len(results) == 2
    detected_langs = {item["lang"] for item in results}
    assert "en" in detected_langs
    assert "fr" in detected_langs


def test_detect_does_not_emit_fasttext_load_model_warning(
    capfd: pytest.CaptureFixture[str],
) -> None:
    """Regression test for issue #11.

    Upstream fasttext.load_model() unconditionally prints a deprecation-style
    warning to stderr on every call. We suppress it inside the wrapper so
    users don't see noise on first use.
    """
    detect_module._models.pop("low_mem", None)
    try:
        detect("hello world", low_memory=True)
    finally:
        captured = capfd.readouterr()

    assert "load_model" not in captured.err, (
        f"fastText's load_model warning leaked to stderr: {captured.err!r}"
    )
