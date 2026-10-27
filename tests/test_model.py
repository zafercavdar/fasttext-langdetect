"""Tests for :mod:`ftlangdetect.model`, the NumPy implementation of fastText.

The prediction tests download the fastText models on first run (cached under
``$FTLANG_CACHE`` or the system temp dir).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ftlangdetect.detect import get_or_load_model
from ftlangdetect.model import FastTextModel, char_ngrams, fasttext_hash


def test_fasttext_hash() -> None:
    # FNV-1a for ASCII; fastText sign-extends bytes >= 0x80.
    assert fasttext_hash(b"a") == 0xE40C292C
    assert fasttext_hash("ğ".encode()) == 1292221156


def test_char_ngrams() -> None:
    ngrams = [g.decode("utf-8") for g in char_ngrams("<ağ>".encode(), 2, 4)]
    assert ngrams == ["<a", "<ağ", "<ağ>", "ağ", "ağ>", "ğ>"]


@pytest.mark.parametrize("payload", [b"", b"<html>404 Not Found</html>", b"\xba\x16O/" + b"\0" * 8])
def test_invalid_model_raises_value_error(tmp_path: Path, payload: bytes) -> None:
    path = tmp_path / "lid.176.bin"
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="not a valid fastText model"):
        FastTextModel(path)


# Reference predictions from fastText itself (fasttext-predict 0.9.2.4).
@pytest.mark.parametrize(
    ("low_memory", "expected"),
    [
        (True, [("fr", 0.846283), ("es", 0.06091), ("en", 0.013187)]),
        (False, [("fr", 0.860397), ("en", 0.026878), ("es", 0.025497)]),
    ],
)
def test_predict_matches_fasttext(low_memory: bool, expected: list[tuple[str, float]]) -> None:
    model = get_or_load_model(low_memory)
    predictions = model.predict("The quick brown fox. Le chat dort sur le canapé.", k=3)
    assert [label for label, _ in predictions] == [label for label, _ in expected]
    for (_, probability), (_, expected_probability) in zip(predictions, expected):
        assert probability == pytest.approx(expected_probability, abs=1e-5)
