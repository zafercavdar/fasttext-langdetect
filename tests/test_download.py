"""Unit tests for the model download helper.

These tests avoid hitting the public model server by monkeypatching
``requests.get`` with an in-memory fake.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest

detect_module = importlib.import_module("ftlangdetect.detect")


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int) -> Any:
        for i in range(0, len(self._payload), chunk_size):
            yield self._payload[i : i + chunk_size]


def test_download_model_caches_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"fake-fasttext-model-bytes" * 1024
    calls: list[str] = []

    def fake_get(url: str, **_: object) -> _FakeResponse:
        calls.append(url)
        return _FakeResponse(payload)

    monkeypatch.setattr(detect_module.requests, "get", fake_get)

    target = detect_module.download_model("lid.176.ftz", cache_dir=tmp_path)

    assert target == tmp_path / "lid.176.ftz"
    assert target.read_bytes() == payload
    assert calls == [f"{detect_module._MODEL_BASE_URL}/lid.176.ftz"]

    # Second call should be served from cache, no extra HTTP request.
    target_again = detect_module.download_model("lid.176.ftz", cache_dir=tmp_path)
    assert target_again == target
    assert calls == [f"{detect_module._MODEL_BASE_URL}/lid.176.ftz"]


def test_download_model_cleans_up_partial_on_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _BoomResponse(_FakeResponse):
        def iter_content(self, chunk_size: int) -> Any:
            yield b"partial"
            raise RuntimeError("network died")

    def fake_get(url: str, **_: object) -> _BoomResponse:
        return _BoomResponse(b"")

    monkeypatch.setattr(detect_module.requests, "get", fake_get)

    with pytest.raises(RuntimeError, match="network died"):
        detect_module.download_model("lid.176.ftz", cache_dir=tmp_path)

    assert not (tmp_path / "lid.176.ftz").exists()
    assert not (tmp_path / "lid.176.ftz.part").exists()


def test_default_cache_dir_respects_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FTLANG_CACHE", str(tmp_path / "custom"))
    assert detect_module._default_cache_dir() == tmp_path / "custom"

    monkeypatch.delenv("FTLANG_CACHE", raising=False)
    assert detect_module._default_cache_dir().name == "fasttext-langdetect"


def test_get_or_load_model_recovers_from_corrupt_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test for issue #15.

    Older releases of this package wrote 4xx/5xx HTML error bodies into the
    model cache because they did not check the HTTP status. Once that file
    was on disk, every subsequent call returned the same corrupt path and
    fasttext raised ``ValueError: ... has wrong file format!``. We now
    detect the load failure, delete the cached file, re-download, and try
    once more.
    """
    detect_module._models.pop("low_mem", None)

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    model_path = cache_dir / "lid.176.ftz"
    model_path.write_bytes(b"<html>404 Not Found</html>")

    download_calls: list[str] = []

    def fake_download(name: str, cache_dir: Path | None = None) -> Path:
        download_calls.append(name)
        if not model_path.exists():
            model_path.write_bytes(b"VALID_FASTTEXT_PAYLOAD")
        return model_path

    load_calls: list[str] = []
    sentinel_model = object()

    def fake_load_model(path: str) -> object:
        load_calls.append(path)
        if Path(path).read_bytes().startswith(b"<html>"):
            msg = f"{path} has wrong file format!"
            raise ValueError(msg)
        return sentinel_model

    monkeypatch.setattr(detect_module, "download_model", fake_download)
    monkeypatch.setattr(detect_module.fasttext, "load_model", fake_load_model)

    try:
        result = detect_module.get_or_load_model(low_memory=True)
    finally:
        detect_module._models.pop("low_mem", None)

    assert result is sentinel_model
    assert download_calls == ["lid.176.ftz", "lid.176.ftz"], (
        "download_model should be called twice: once for the corrupt cache, once after deletion"
    )
    assert len(load_calls) == 2, "load_model should be retried after corruption"
    assert model_path.exists(), "the freshly downloaded model must be present"
    assert model_path.read_bytes() == b"VALID_FASTTEXT_PAYLOAD"


def test_get_or_load_model_propagates_second_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the freshly downloaded model also fails to load, surface the error."""
    detect_module._models.pop("high_mem", None)

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    model_path = cache_dir / "lid.176.bin"
    model_path.write_bytes(b"still-broken")

    def fake_download(name: str, cache_dir: Path | None = None) -> Path:
        model_path.write_bytes(b"still-broken")
        return model_path

    def always_fail(path: str) -> object:
        msg = f"{path} has wrong file format!"
        raise ValueError(msg)

    monkeypatch.setattr(detect_module, "download_model", fake_download)
    monkeypatch.setattr(detect_module.fasttext, "load_model", always_fail)

    try:
        with pytest.raises(ValueError, match="wrong file format"):
            detect_module.get_or_load_model(low_memory=False)
    finally:
        detect_module._models.pop("high_mem", None)
