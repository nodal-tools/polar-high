"""Pins the vectorized ``Log2HistogramReducer.update`` segmented reduction.

Three properties are pinned:

(a) EQUIVALENCE — on a random batch with several distinct col_ids mapping
    to a few buckets (plus some col_ids classifying to ``None``), the
    reducer's per-bucket ``(count, abs_min, abs_max)`` are BYTE-identical
    to an independent per-row Python groupby reference and the log2-sum
    matches within ``rtol=1e-9`` (FP reassociation only).

(b) DROP semantics — ``None`` classifications, an all-``None`` batch, an
    empty batch and an all-masked-out batch all behave exactly as the old
    per-uniq loop did (entry dropped / clean early-return).

(c) BATCH-SPLIT INVARIANCE — feeding the same data in one batch vs split
    into several batches yields byte-identical count/min/max (those
    combine exactly; the reducer's batch combine is order-free).
"""

from __future__ import annotations

import math

import numpy as np

from polar_high.autoscale._coef_walk import Log2HistogramReducer, _scaled_abs


def _make_case(n: int, n_uniq: int, n_buckets: int, seed: int):
    """Random batch: int col_ids in ``[0, n_uniq)``, magnitudes spanning
    many decades, a ``classify`` that sends every 37th distinct col_id to
    ``None`` and the rest to one of ``n_buckets`` string keys.  ``rid`` is
    the column-spine sentinel (-1) so ``_scaled_abs`` with a scalar-only
    scale returns ``|coef|`` unchanged.  A few entries are forced to zero /
    negative / non-finite to exercise the finite & positive guard."""
    rng = np.random.default_rng(seed)
    cid = rng.integers(0, n_uniq, size=n).astype(np.int64)
    coef = (10.0 ** rng.uniform(-6.0, 6.0, size=n)).astype(np.float64)
    if n >= 10:
        coef[0] = 0.0  # masked out (a == 0)
        coef[1] = -coef[1]  # negative coef -> abs positive, KEPT
        coef[2] = np.inf  # masked out (non-finite)
        coef[3] = np.nan  # masked out (non-finite)
    rid = np.full(n, -1, dtype=np.int64)
    bmap = {c: (None if c % 37 == 0 else f"B{c % n_buckets}") for c in range(n_uniq)}

    def classify(c: int):
        return bmap.get(int(c))

    return rid, cid, coef, classify


def _reference(rid, cid, coef, classify, scale):
    """Independent per-row Python groupby reference for the histogram.

    Mirrors the reducer guards exactly: scale via ``_scaled_abs``, drop
    non-finite / non-positive magnitudes, drop ``None`` classifications,
    then per bucket accumulate ``(sum log2, count, min, max)``."""
    a = _scaled_abs(rid, cid, coef, scale)
    acc: dict[object, list] = {}
    for i in range(a.size):
        v = float(a[i])
        if not math.isfinite(v) or v <= 0.0:
            continue
        bk = classify(int(cid[i]))
        if bk is None:
            continue
        lg = math.log2(v)
        if bk not in acc:
            acc[bk] = [lg, 1, v, v]
        else:
            e = acc[bk]
            e[0] += lg
            e[1] += 1
            if v < e[2]:
                e[2] = v
            if v > e[3]:
                e[3] = v
    return {k: tuple(v) for k, v in acc.items()}


def test_hist_vectorized_matches_reference():
    scale = (None, 0, None)
    for n, n_uniq, n_buckets, seed in [
        (250_000, 250_000, 1, 1),
        (250_000, 250_000, 5, 2),
        (250_000, 60_000, 3, 3),
        (250_000, 1_000, 4, 4),
        (200_003, 199_991, 6, 5),
    ]:
        rid, cid, coef, classify = _make_case(n, n_uniq, n_buckets, seed)
        red = Log2HistogramReducer(scale, classify)
        red.init()
        red.update(rid, cid, coef)
        got = red.finalize()
        ref = _reference(rid, cid, coef, classify, scale)

        assert set(got) == set(ref), (n, set(got), set(ref))
        # Sanity: the case actually produced buckets (not a trivial pass).
        assert got, (n, "no buckets produced")
        for k in ref:
            g_slog, g_cnt, g_min, g_max = got[k]
            r_slog, r_cnt, r_min, r_max = ref[k]
            # count / min / max must be BYTE-identical (order-free).
            assert g_cnt == r_cnt, (n, k, g_cnt, r_cnt)
            assert g_min == r_min, (n, k, g_min, r_min)
            assert g_max == r_max, (n, k, g_max, r_max)
            # log2-sum: FP-reassociation only.
            assert math.isclose(g_slog, r_slog, rel_tol=1e-9, abs_tol=0.0), (n, k, g_slog, r_slog)


def test_hist_all_none_classification_drops_everything():
    """Every col_id classifies to ``None`` -> no buckets folded (the
    ``nb == 0`` early-return), matching the old loop's ``continue`` skip."""
    scale = (None, 0, None)
    rng = np.random.default_rng(11)
    n = 5000
    cid = rng.integers(0, 200, size=n).astype(np.int64)
    coef = (rng.random(n) * 1e3 + 1e-3).astype(np.float64)
    rid = np.full(n, -1, dtype=np.int64)

    red = Log2HistogramReducer(scale, lambda c: None)
    red.init()
    red.update(rid, cid, coef)
    assert red.finalize() == {}


def test_hist_empty_and_all_masked_batches_early_return():
    """Empty batch and an all-non-positive batch both early-return cleanly
    (no buckets, no crash) — matching the original guards."""
    scale = (None, 0, None)

    def classify(c):
        return f"B{int(c) % 3}"

    red = Log2HistogramReducer(scale, classify)
    red.init()
    # Empty.
    red.update(
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.float64),
    )
    assert red.finalize() == {}

    # All-masked-out (every coef zero -> magnitude not > 0).
    n = 100
    red.update(
        np.full(n, -1, dtype=np.int64),
        np.arange(n, dtype=np.int64),
        np.zeros(n, dtype=np.float64),
    )
    assert red.finalize() == {}


def test_hist_batch_split_invariant():
    """count/min/max are invariant to how the stream is split into batches;
    the log2-sum matches the single-batch result within rtol=1e-9 (the
    reducer's batch combine is order-free for count/min/max)."""
    scale = (None, 0, None)
    rid, cid, coef, classify = _make_case(180_000, 50_000, 4, seed=7)

    whole = Log2HistogramReducer(scale, classify)
    whole.init()
    whole.update(rid, cid, coef)
    ref = whole.finalize()

    split = Log2HistogramReducer(scale, classify)
    split.init()
    bounds = [0, 1, 40_000, 40_001, 120_000, rid.size]
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        split.update(rid[lo:hi], cid[lo:hi], coef[lo:hi])
    got = split.finalize()

    assert set(got) == set(ref)
    assert got, "no buckets produced"
    for k in ref:
        r_slog, r_cnt, r_min, r_max = ref[k]
        g_slog, g_cnt, g_min, g_max = got[k]
        assert g_cnt == r_cnt, (k, g_cnt, r_cnt)
        assert g_min == r_min, (k, g_min, r_min)
        assert g_max == r_max, (k, g_max, r_max)
        assert math.isclose(g_slog, r_slog, rel_tol=1e-9, abs_tol=0.0), (k, g_slog, r_slog)
