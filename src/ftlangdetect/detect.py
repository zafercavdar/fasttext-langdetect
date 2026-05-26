"""Language detection backed by Facebook's pre-trained fastText ``lid.176`` model."""

from __future__ import annotations

import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict, overload

import fasttext
import requests

if TYPE_CHECKING:
    from fasttext.FastText import _FastText

__all__ = ["DetectionResult", "detect", "get_or_load_model"]

logger = logging.getLogger(__name__)

_MODEL_BASE_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models"
_LOW_MEM_MODEL = "lid.176.ftz"
_HIGH_MEM_MODEL = "lid.176.bin"
_DOWNLOAD_TIMEOUT_S = 60
_DOWNLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MiB

_models: dict[str, _FastText] = {}
_model_lock = threading.Lock()


def _default_cache_dir() -> Path:
    """Resolve the directory used to cache downloaded models."""
    override = os.environ.get("FTLANG_CACHE")
    if override:
        return Path(override).expanduser()
    return Path(tempfile.gettempdir()) / "fasttext-langdetect"


class DetectionResult(TypedDict):
    """The structured result returned by :func:`detect`."""

    lang: str
    score: float


def download_model(name: str, cache_dir: Path | None = None) -> Path:
    """Download a fastText model to the local cache and return its path.

    If the model is already cached the cached copy is returned and no
    network request is made.

    Args:
        name: Model filename, e.g. ``lid.176.bin`` or ``lid.176.ftz``.
        cache_dir: Directory used to store the model. Defaults to
            ``$FTLANG_CACHE`` or the system temp directory.

    Returns:
        Absolute path to the model on disk.
    """
    cache_dir = (cache_dir or _default_cache_dir()).expanduser()
    target_path = cache_dir / name
    if target_path.exists():
        return target_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    url = f"{_MODEL_BASE_URL}/{name}"
    logger.info("Downloading fastText model %s from %s", name, url)

    # Stream into a temp file in the same directory so the move is atomic.
    tmp_path = target_path.with_suffix(target_path.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=_DOWNLOAD_TIMEOUT_S) as response:
            response.raise_for_status()
            with tmp_path.open("wb") as fp:
                for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK_SIZE):
                    if chunk:
                        fp.write(chunk)
        tmp_path.replace(target_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    logger.info("Downloaded fastText model to %s", target_path)
    return target_path


def get_or_load_model(low_memory: bool = False) -> _FastText:
    """Return the cached fastText model, loading it from disk on first use.

    If the cached file fails to load (for example because an older version
    of this package wrote a partial or 4xx/5xx HTML response to disk under
    the model filename — see issue #15) the corrupt file is removed and a
    fresh copy is downloaded automatically, then loaded once more.

    Args:
        low_memory: If ``True``, use the compressed ``lid.176.ftz`` model
            (slightly lower accuracy, much smaller memory footprint).

    Returns:
        The loaded fastText model instance.

    Raises:
        ValueError: If a freshly downloaded model still fails to load,
            which usually indicates a problem upstream rather than in
            the local cache.
    """
    key = "low_mem" if low_memory else "high_mem"
    model = _models.get(key)
    if model is not None:
        return model

    with _model_lock:
        model = _models.get(key)
        if model is not None:
            return model

        model_name = _LOW_MEM_MODEL if low_memory else _HIGH_MEM_MODEL
        model_path = download_model(model_name)
        try:
            model = fasttext.load_model(str(model_path))
        except ValueError as exc:
            logger.warning(
                "Cached fastText model at %s failed to load (%s); "
                "removing it and re-downloading once.",
                model_path,
                exc,
            )
            model_path.unlink(missing_ok=True)
            model_path = download_model(model_name)
            model = fasttext.load_model(str(model_path))

        _models[key] = model
        return model


@overload
def detect(text: str, low_memory: bool = ..., k: Literal[1] = ...) -> DetectionResult: ...
@overload
def detect(text: str, low_memory: bool = ..., *, k: int) -> list[DetectionResult]: ...
def detect(
    text: str,
    low_memory: bool = False,
    k: int = 1,
) -> DetectionResult | list[DetectionResult]:
    """Detect the language of ``text`` using fastText's ``lid.176`` model.

    Args:
        text: Input text. Must be a single line (no embedded newlines)
            and UTF-8 decodable.
        low_memory: If ``True``, use the compressed model.
        k: Number of language candidates to return. When ``k == 1``
            (the default), a single :class:`DetectionResult` is returned,
            preserving backward compatibility. When ``k > 1``, a list of
            up to ``k`` :class:`DetectionResult` items is returned, sorted
            by ``score`` in descending order. Useful for bilingual or
            code-switched text.

    Returns:
        A :class:`DetectionResult` when ``k == 1``, or a list of
        :class:`DetectionResult` (length up to ``k``, sorted by descending
        score) when ``k > 1``.

    Raises:
        ValueError: If ``text`` contains newline characters, which fastText
            does not accept in :py:meth:`fasttext.FastText._FastText.predict`,
            or if ``k < 1``.
    """
    if "\n" in text:
        msg = "detect() does not support newline characters in text; split into lines first."
        raise ValueError(msg)
    if k < 1:
        msg = f"k must be >= 1, got {k}"
        raise ValueError(msg)

    model = get_or_load_model(low_memory)

    # Call the pybind11 layer directly to avoid fasttext 0.9.x's call to
    # ``np.array(probs, copy=False)``, which raises under NumPy >= 2.0.
    # See https://github.com/zafercavdar/fasttext-langdetect/issues/17.
    predictions = model.f.predict(text + "\n", k, 0.0, "strict")
    if not predictions:
        msg = "fastText returned no prediction for the given text."
        raise RuntimeError(msg)

    results: list[DetectionResult] = [
        DetectionResult(
            lang=label.replace("__label__", ""),
            score=min(float(probability), 1.0),
        )
        for probability, label in predictions
    ]

    if k == 1:
        return results[0]
    return results
