"""fastText's ``lid.176`` models, run with NumPy.

Reads fastText's binary model format and reproduces fastText's predictions
for supervised models that use hierarchical softmax and no word n-grams,
which is what both ``lid.176.bin`` and ``lid.176.ftz`` are. The ``.ftz``
model is also pruned and product-quantized; its quantized matrices are
decoded into dense ones on load.
"""

from __future__ import annotations

import re
import struct
from collections.abc import Iterator
from pathlib import Path

import numpy as np

__all__ = ["FastTextModel"]

_FASTTEXT_MAGIC = 793712314
_FASTTEXT_VERSION = 12
_LOSS_HS = 1
_EOS = b"</s>"
_LABEL_PREFIX = b"__label__"
# Centroids per sub-quantizer; fastText's product quantizer uses 8-bit codes.
_PQ_KSUB = 256
# Bytes fastText splits words on (Dictionary::readWord).
_FASTTEXT_SPACE_RE = re.compile(rb"[ \n\r\t\v\f\0]+")


def fasttext_hash(data: bytes) -> int:
    """fastText's FNV-1a hash, which sign-extends each byte (Dictionary::hash)."""
    h = 2166136261
    for b in data:
        h ^= b if b < 0x80 else b | 0xFFFFFF00
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def char_ngrams(word: bytes, minn: int, maxn: int) -> Iterator[bytes]:
    """Character n-grams of ``word`` (UTF-8 bytes), as Dictionary::computeSubwords."""
    size = len(word)
    for i in range(size):
        if word[i] & 0xC0 == 0x80:
            continue
        j = i
        n = 1
        while j < size and n <= maxn:
            j += 1
            while j < size and word[j] & 0xC0 == 0x80:
                j += 1
            if n >= minn and not (n == 1 and (i == 0 or j == size)):
                yield word[i:j]
            n += 1


def _read_product_quantizer(data: bytes, pos: int, codes: np.ndarray) -> tuple[np.ndarray, int]:
    """Read a ProductQuantizer and decode ``codes`` (one row per vector) with it."""
    dim, nsubq, dsub, lastdsub = struct.unpack_from("<4i", data, pos)
    pos += 16
    centroids = np.frombuffer(data, dtype="<f4", count=dim * _PQ_KSUB, offset=pos)
    pos += 4 * dim * _PQ_KSUB
    parts = []
    for q in range(nsubq):
        d = lastdsub if q == nsubq - 1 else dsub
        start = q * _PQ_KSUB * dsub
        table = centroids[start : start + _PQ_KSUB * d].reshape(_PQ_KSUB, d)
        parts.append(table[codes[:, q]])
    return np.hstack(parts), pos


def _read_matrix(data: bytes, pos: int) -> tuple[np.ndarray, int]:
    """Read a DenseMatrix or QuantMatrix, preceded by its "quantized" flag."""
    quantized = data[pos]
    pos += 1
    if not quantized:
        m, n = struct.unpack_from("<qq", data, pos)
        pos += 16
        matrix = np.frombuffer(data, dtype="<f4", count=m * n, offset=pos)
        return matrix.reshape(m, n), pos + 4 * m * n

    qnorm, m, _, codesize = struct.unpack_from("<?qqi", data, pos)
    pos += 21
    codes = np.frombuffer(data, dtype=np.uint8, count=codesize, offset=pos).reshape(m, -1)
    pos += codesize
    matrix, pos = _read_product_quantizer(data, pos, codes)
    if qnorm:
        norm_codes = np.frombuffer(data, dtype=np.uint8, count=m, offset=pos).reshape(m, 1)
        pos += m
        norms, pos = _read_product_quantizer(data, pos, norm_codes)
        matrix *= norms
    return matrix, pos


class FastTextModel:
    """A supervised fastText model with hierarchical softmax loss."""

    def __init__(self, path: Path) -> None:
        data = Path(path).read_bytes()
        try:
            self._load(data)
        except (struct.error, ValueError, IndexError) as exc:
            msg = f"{path} is not a valid fastText model: {exc}"
            raise ValueError(msg) from exc

    def _load(self, data: bytes) -> None:
        magic, version = struct.unpack_from("<ii", data, 0)
        if magic != _FASTTEXT_MAGIC or version != _FASTTEXT_VERSION:
            msg = "unsupported file format"
            raise ValueError(msg)
        args = struct.unpack_from("<12id", data, 8)
        dim, _, _, _, _, word_ngrams, loss, _, self.bucket, self.minn, self.maxn = args[:11]
        if loss != _LOSS_HS or word_ngrams != 1:
            msg = "only hierarchical softmax models without word n-grams are supported"
            raise ValueError(msg)

        pos = 64
        size, nwords, nlabels, _, pruneidx_size = struct.unpack_from("<iiiqq", data, pos)
        pos += 28
        self.word_ids: dict[bytes, int] = {}
        self.labels: list[str] = []
        counts = []
        for i in range(size):
            end = data.index(b"\0", pos)
            entry = data[pos:end]
            count, entry_type = struct.unpack_from("<qb", data, end + 1)
            pos = end + 10
            if entry_type == 0:
                self.word_ids[entry] = i
            else:
                self.labels.append(entry[len(_LABEL_PREFIX) :].decode("utf-8"))
                counts.append(count)
        self.nwords = nwords

        # A pruned model keeps only some n-gram buckets, renumbered from 0.
        # pruneidx_size is -1 when the model is not pruned.
        self.pruneidx: dict[int, int] | None = None
        if pruneidx_size >= 0:
            pairs = struct.unpack_from(f"<{2 * pruneidx_size}i", data, pos)
            self.pruneidx = dict(zip(pairs[::2], pairs[1::2]))
            pos += 8 * pruneidx_size

        self.input, pos = _read_matrix(data, pos)
        self.output, pos = _read_matrix(data, pos)
        if self.input.shape[1] != dim or len(self.labels) != nlabels:
            msg = "inconsistent dimensions"
            raise ValueError(msg)
        self._build_tree(counts)

    def _build_tree(self, counts: list[int]) -> None:
        """Build the Huffman tree of the labels (HierarchicalSoftmaxLoss::buildTree).

        Stores, for each label, which internal nodes are on its path from the
        root and whether the path goes left or right at each of them.
        """
        osz = len(counts)
        count = list(counts) + [1e15] * (osz - 1)
        parent = [-1] * (2 * osz - 1)
        right = [False] * (2 * osz - 1)
        leaf, node = osz - 1, osz
        for i in range(osz, 2 * osz - 1):
            mini = []
            for _ in range(2):
                if leaf >= 0 and count[leaf] < count[node]:
                    mini.append(leaf)
                    leaf -= 1
                else:
                    mini.append(node)
                    node += 1
            count[i] = count[mini[0]] + count[mini[1]]
            parent[mini[0]] = parent[mini[1]] = i
            right[mini[1]] = True

        self.goes_left = np.zeros((osz, osz - 1), dtype=np.float32)
        self.goes_right = np.zeros((osz, osz - 1), dtype=np.float32)
        for label in range(osz):
            node = label
            while parent[node] != -1:
                path = self.goes_right if right[node] else self.goes_left
                path[label, parent[node] - osz] = 1
                node = parent[node]

    def _subword_ids(self, word: bytes) -> list[int]:
        """Input rows of a word: its own row if known, and its character n-grams."""
        ids = []
        wid = self.word_ids.get(word, -1)
        if wid >= 0:
            ids.append(wid)
        if word != _EOS:
            for ngram in char_ngrams(b"<" + word + b">", self.minn, self.maxn):
                h = fasttext_hash(ngram) % self.bucket
                if self.pruneidx is None:
                    ids.append(self.nwords + h)
                elif h in self.pruneidx:
                    ids.append(self.nwords + self.pruneidx[h])
        return ids

    def _line_ids(self, line: str) -> list[int]:
        """Input rows of a line of text (Dictionary::getLine)."""
        words = [*_FASTTEXT_SPACE_RE.split(line.encode("utf-8")), _EOS]
        ids = []
        for word in words:
            # fastText reads words that look like labels as labels.
            if word and not word.startswith(_LABEL_PREFIX):
                ids.extend(self._subword_ids(word))
        return ids

    def predict(self, line: str, k: int = 1, threshold: float = 0.0) -> list[tuple[str, float]]:
        """Return up to ``k`` ``(label, probability)`` pairs, the most probable first."""
        ids = self._line_ids(line)
        if not ids:
            return []
        hidden = self.input[ids].mean(axis=0, dtype=np.float32)
        # NumPy 2.0/2.1 on macOS arm64 (Accelerate) emits spurious floating-point
        # warnings from matmul, and exp() may overflow to inf, which is harmless.
        with np.errstate(all="ignore"):
            f = 1.0 / (1.0 + np.exp(-(self.output[: len(self.labels) - 1] @ hidden)))
            scores = self.goes_left @ np.log(1.0 - f + 1e-5) + self.goes_right @ np.log(f + 1e-5)
        best = np.argsort(-scores, kind="stable")[:k]
        return [
            (self.labels[i], float(np.exp(scores[i])))
            for i in best
            if scores[i] >= np.log(threshold + 1e-5)
        ]
