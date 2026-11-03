"""``Log2HistogramReducer`` fed an ``ArrayClassifier`` equals the callable form.

The array form must give the reducer exactly the callable's accumulator:
same keys, same dict insertion order and bit-identical
``(sum log2, count, min, max)`` per bucket, over several batches, with
columns that have no bucket (``-1``), col_ids past the end of the table
and keys that compare equal (folded into one bucket, as the callable
folds them).
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from polar_high.autoscale._coef_walk import ArrayClassifier, Log2HistogramReducer


class _Key:
    """Hashable key whose equality is by name, so two distinct objects can
    compare equal (the case that must share one bucket)."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Key) and other.name == self.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:
        return f"_Key({self.name!r})"


def _bits(acc: dict) -> list:
    """Accumulator as (key, exact bit patterns) in insertion order."""
    out = []
    for k, (slog, cnt, amin, amax) in acc.items():
        out.append(
            (k, struct.pack("<d", slog), cnt, struct.pack("<d", amin), struct.pack("<d", amax))
        )
    return out


def _run(classify, batches, scale) -> dict:
    red = Log2HistogramReducer(scale, classify)
    red.init()
    for rid, cid, coef in batches:
        red.update(rid, cid, coef)
    return red.finalize()


def _batches(rng, n_cols: int, n_batches: int, batch_n: int, overshoot: int):
    out = []
    for _ in range(n_batches):
        cid = rng.integers(0, n_cols + overshoot, size=batch_n).astype(np.int64)
        coef = (10.0 ** rng.uniform(-8.0, 8.0, size=batch_n)) * rng.choice(
            [-1.0, 1.0], size=batch_n
        )
        coef[rng.random(batch_n) < 0.01] = 0.0
        coef[rng.random(batch_n) < 0.005] = np.nan
        out.append((np.full(batch_n, -1, dtype=np.int64), cid, coef))
    return out


@pytest.mark.parametrize("seed", range(8))
def test_array_classifier_bit_identical_to_callable(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n_cols = int(rng.integers(50, 5000))
    names = ["a", "b", "c", "d", "e", "b"]  # last key equals the second
    keys = [_Key(n) for n in names]
    # Randomly permute so the first-appearance order differs from the key order.
    perm = rng.permutation(len(keys))
    keys = [keys[i] for i in perm]
    bucket = rng.integers(-1, len(keys), size=n_cols).astype(np.int16)
    clf = ArrayClassifier(bucket, keys)

    def classify(c: int):
        if c >= n_cols:
            return None
        b = int(bucket[c])
        return None if b < 0 else keys[b]

    batches = _batches(
        rng,
        n_cols,
        n_batches=int(rng.integers(1, 6)),
        batch_n=int(rng.integers(1, 20000)),
        overshoot=int(rng.integers(0, 50)),
    )
    for scale in [(None, 0, None), (None, 0, 10.0 ** rng.uniform(-3, 3, size=n_cols + 60))]:
        want = _run(classify, batches, scale)
        got = _run(clf, batches, scale)
        assert want, "case produced no buckets"
        assert _bits(got) == _bits(want)
        # The kept key object is the callable's object for the folded bucket.
        assert [id(k) for k in got] == [id(k) for k in want]


def test_array_classifier_all_unbucketed_and_empty() -> None:
    clf = ArrayClassifier(np.full(10, -1, dtype=np.int32), ["x"])
    rid = np.full(5, -1, dtype=np.int64)
    assert _run(clf, [(rid, np.arange(5, dtype=np.int64), np.ones(5))], (None, 0, None)) == {}
    empty = ArrayClassifier(np.empty(0, dtype=np.int32), [])
    assert _run(empty, [(rid, np.arange(5, dtype=np.int64), np.ones(5))], (None, 0, None)) == {}


def test_array_classifier_call_and_validation() -> None:
    clf = ArrayClassifier(np.array([1, -1, 0], dtype=np.int8), ["p", "q"])
    assert [clf(0), clf(1), clf(2), clf(3)] == ["q", None, "p", None]
    with pytest.raises(ValueError):
        ArrayClassifier(np.array([2]), ["p", "q"])
    with pytest.raises(ValueError):
        ArrayClassifier(np.array([-2]), ["p"])
    with pytest.raises(TypeError):
        ArrayClassifier(np.array([0.0]), ["p"])
