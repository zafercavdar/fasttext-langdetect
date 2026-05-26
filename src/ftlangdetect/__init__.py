"""fastText-based language detection."""

from importlib.metadata import PackageNotFoundError, version

from ftlangdetect.detect import DetectionResult, detect

try:
    __version__ = version("fasttext-langdetect")
except PackageNotFoundError:  # pragma: no cover - only happens in editable dev installs
    __version__ = "0.0.0+unknown"

__all__ = ["DetectionResult", "__version__", "detect"]
