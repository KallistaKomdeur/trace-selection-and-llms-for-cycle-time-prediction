"""Trace selection with the combined control-flow / temporal distance of Section 3.1.

    d_cf(s1, s2) = DL(act(s1), act(s2)) / max(|s1|, |s2|)
    d_t (s1, s2) = |et(s1) - et(s2)| / max(et(s1), et(s2), 1)      et = elapsed time of the prefix
    d_lambda     = lambda * d_cf + (1 - lambda) * d_t,  similarity = 1 - d_lambda

For a running prefix of k events the candidates are the prefixes of the training traces
(exactly length k by default, or every length <= k with candidates="upto"); the n most
similar prefixes are returned as their *completed* traces. lambda=1 reduces to the
control-flow-only selection of similar_prefix_sets.py.
"""
import random
from copy import deepcopy

from utils.similar_prefix_sets import _edit_distance, _select_test_cases_randomly


def _acts(seq, k):
    return tuple(e[0] if isinstance(e, (list, tuple)) else str(e) for e in seq[:k])


def _elapsed(seq, k):
    """Elapsed time of the k-prefix: cumulative minutes at its last event."""
    return float(seq[k - 1][1]) if k and len(seq) >= k else 0.0


def d_cf(acts1, acts2):
    return _edit_distance(acts1, acts2) / max(len(acts1), len(acts2), 1)


def d_t(et1, et2):
    return abs(et1 - et2) / max(et1, et2, 1.0)


def d_lambda(acts1, et1, acts2, et2, lam):
    # The endpoints skip the term whose weight is zero: identical value, and at lambda=0
    # it avoids an edit distance per candidate, which dominates the cost on long traces.
    if lam == 0.0:
        return d_t(et1, et2)
    if lam == 1.0:
        return d_cf(acts1, acts2)
    return lam * d_cf(acts1, acts2) + (1.0 - lam) * d_t(et1, et2)


def retrieve(test_seq, k, train_cases, n, lam, candidates="equal", rng=None):
    """Return the n training cases whose k-prefix is closest to the test prefix.

    Returns a list of (distance, case) sorted by distance; ties are shuffled with `rng`
    before the cut, as the control-flow-only implementation does.
    """
    t_acts, t_et = _acts(test_seq, k), _elapsed(test_seq, k)
    rng = rng or random
    # distance depends only on the candidate prefix, so it is computed once per distinct
    # (activities, elapsed) pair and shared by every case with that prefix
    groups = {}
    for case in train_cases:
        seq = case.get("ActTimeSeq", [])
        lengths = [k] if candidates == "equal" else range(1, min(k, len(seq)) + 1)
        for m in lengths:
            if len(seq) < m:
                continue
            key = (_acts(seq, m), _elapsed(seq, m))
            groups.setdefault(key, []).append(case)
    scored = sorted((d_lambda(t_acts, t_et, a, et, lam), key) for key, (a, et) in
                    ((key, key) for key in groups))
    selected, seen, i = [], set(), 0
    while i < len(scored) and len(selected) < n:
        d = scored[i][0]
        tied = []
        while i < len(scored) and scored[i][0] == d:
            tied.extend(groups[scored[i][1]])
            i += 1
        rng.shuffle(tied)
        for c in tied:
            cid = id(c)
            if cid not in seen and len(selected) < n:
                seen.add(cid)
                selected.append((d, c))
    return selected


def generate_lambda_sets(train_cases, test_cases, n_sets, examples_count, lam, seed,
                         candidates="equal"):
    """Same protocol as generate_similar_prefix_sets, with d_lambda as the distance."""
    tests = _select_test_cases_randomly(test_cases, n_sets, seed)
    rng = random.Random(seed + 1)
    out = []
    for rec in tests:
        k, tc = rec["prefix_len"], rec["test_case"]
        sel = retrieve(tc["ActTimeSeq"], k, train_cases, examples_count, lam, candidates, rng)
        out.append({"examples": [deepcopy(c) for _, c in sel], "test_case": tc,
                    "prefix_length": k, "total_case_length": rec["case_len"],
                    "mode_truncated_training": False, "lambda": lam,
                    "distances": [d for d, _ in sel]})
    return out
