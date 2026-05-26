"""Integration tests for :mod:`ftlangdetect.detect`.

These tests download the fastText model on first run (cached under
``$FTLANG_CACHE`` or the system temp dir).
"""

from __future__ import annotations

import pytest

from ftlangdetect import DetectionResult, detect


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


def test_detect_rejects_newlines() -> None:
    with pytest.raises(ValueError, match="newline"):
        detect("line one\nline two", low_memory=True)
