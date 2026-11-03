"""``_dedup_sum_coo``: duplicate (row, col) coefficients are summed in their production order, so the result is
the same bits whatever the order a hash group-by would visit them in."""

import numpy as np

from polar_high.engine import _dedup_sum_coo


def test_three_term_group_sums_in_production_order() -> None:
    # +50, −0.005, −50 on one cell: production-order sum is (50 − 0.005) − 50, not exactly −0.005.
    r = np.array([0, 1, 0, 0], dtype=np.int64)
    c = np.array([3, 2, 3, 3], dtype=np.int64)
    v = np.array([50.0, 7.0, -0.005, -50.0])
    rr, cc, vv = _dedup_sum_coo(r, c, v)
    assert rr.tolist() == [0, 1] and cc.tolist() == [3, 2]
    assert vv[0] == (50.0 + -0.005) + -50.0 and vv[1] == 7.0


def test_result_independent_of_unrelated_rows() -> None:
    """Moving other cells' triples around (keeping each cell's own order) gives the same bits."""
    rng = np.random.default_rng(0)
    r = rng.integers(0, 5, 200)
    c = rng.integers(0, 5, 200)
    v = rng.normal(size=200)
    rr, cc, vv = _dedup_sum_coo(r, c, v)
    order = np.lexsort((c, r))  # groups contiguous, each group's own order kept
    r2, c2, v2 = _dedup_sum_coo(r[order], c[order], v[order])
    assert rr.tolist() == r2.tolist() and cc.tolist() == c2.tolist() and vv.tolist() == v2.tolist()
    for i in range(rr.size):
        assert np.isclose(vv[i], v[(r == rr[i]) & (c == cc[i])].sum(), rtol=1e-12, atol=1e-15)
    assert len(set(zip(rr.tolist(), cc.tolist()))) == rr.size


def test_empty() -> None:
    e = np.zeros(0, dtype=np.int64)
    rr, cc, vv = _dedup_sum_coo(e, e, np.zeros(0))
    assert rr.size == cc.size == vv.size == 0
