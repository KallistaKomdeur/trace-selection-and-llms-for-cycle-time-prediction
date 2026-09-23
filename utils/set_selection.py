import random
import time
from collections import defaultdict
from copy import deepcopy

import pandas as pd

def edit_distance(seq_a, seq_b):
    """Damerau-Levenshtein distance (OSA var) between two activity sequences"""
    m, n = len(seq_a), len(seq_b)
    two_back = list(range(n + 1))   # row i-2
    one_back = list(range(n + 1))   # row i-1
    current = [0] * (n + 1)         # row i

    for i in range(1, m + 1):
        current[0] = i
        for j in range(1, n + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            current[j] = min(
                one_back[j] + 1,        # deletion
                current[j - 1] + 1,     # insertion
                one_back[j - 1] + cost, # substitution
            )
            if i > 1 and j > 1 and seq_a[i - 1] == seq_b[j - 2] and seq_a[i - 2] == seq_b[j - 1]:
                current[j] = min(current[j], two_back[j - 2] + 1)  # transposition
        two_back, one_back, current = one_back, current, two_back

    return one_back[n]

def normalized_edit_distance(seq_a, seq_b):
    """Raw DL/OSA distance divided by the length of the longer of the two sequences,
    so distances are comparable across prefixes of different lengths."""
    dist = edit_distance(seq_a, seq_b)
    denom = max(len(seq_a), len(seq_b), 1)
    return dist / denom

def extract_prefix_activities(case, prefix_len):
    """Gets the control-flow variant (activity names only) of a case's prefix."""
    seq = case.get("ActTimeSeq", [])
    return tuple(e[0] if isinstance(e, (list, tuple)) else str(e) for e in seq[:prefix_len])

def prepare_test_cases(test_cases, seed):
    """Prepares every eligible test case with a random prefix length in [2, case_len - 1]"""
    random.seed(seed)

    eligible = [c for c in test_cases if len(c.get("ActTimeSeq", [])) >= 3]

    result = []
    for test_case_full in eligible:
        seq = test_case_full.get("ActTimeSeq", [])
        case_len = len(seq)
        prefix_len = random.randint(2, case_len - 1)

        truncated_test = deepcopy(test_case_full)
        truncated_test["ActTimeSeq"] = seq[:prefix_len]
        truncated_test["true_total_time"] = test_case_full.get("total_time")
        truncated_test["total_time"] = "RUNNING"
        truncated_test["true_total_length"] = case_len

        result.append({"test_case": truncated_test, "prefix_len": prefix_len, "case_len": case_len})

    return result

def retrieve_random_train_cases(train_cases, examples_count):
    """Baseline retrieval: training examples sampled uniformly at random"""
    n = min(examples_count, len(train_cases))
    selected = random.sample(train_cases, n)
    return [deepcopy(c) for c in selected]

def build_train_variants(train_cases, prefix_len):
    """Groups training cases by their prefix control-flow variant, so distance to
    the test prefix is computed once per variant rather than once per case."""
    variants = defaultdict(list)
    for case in train_cases:
        if len(case.get("ActTimeSeq", [])) < prefix_len:
            continue
        variants[extract_prefix_activities(case, prefix_len)].append(case)
    return variants

def retrieve_similar_train_cases(truncated_test, prefix_len, train_cases, examples_count):
    """Picks the training cases whose prefix variant has the smallest normalized
    DL/OSA distance to the test prefix, with ties broken randomly."""
    test_activities = extract_prefix_activities(truncated_test, prefix_len)
    variants = build_train_variants(train_cases, prefix_len)
    scored_variants = sorted(((normalized_edit_distance(test_activities, variant), variant) for variant in variants), key=lambda x: x[0])

    selected = []
    i = 0
    while i < len(scored_variants) and len(selected) < examples_count:
        dist = scored_variants[i][0]

        tied_variants = []
        while i < len(scored_variants) and scored_variants[i][0] == dist:
            tied_variants.append(scored_variants[i][1])
            i += 1

        tied_cases = [case for variant in tied_variants for case in variants[variant]]
        random.shuffle(tied_cases)

        remaining = examples_count - len(selected)
        selected.extend(tied_cases[:remaining])

    return [deepcopy(c) for c in selected]

def prefix_cycle_time_seconds(case, prefix_len):
    """Gets the timestamp of the final event in the prefix"""
    seq = case.get("ActTimeSeq", [])[:prefix_len]
    if len(seq) < 1:
        return 0.0
    return pd.to_numeric(seq[-1][1])

def normalized_cycle_time_distance(cycle_time_a, cycle_time_b):
    """Absolute difference between two prefix cycle times, normalized by the larger
    of the two so the result is on a comparable [0, 1]-ish scale to the DL distance."""
    denom = max(abs(cycle_time_a), abs(cycle_time_b), 1.0)
    return abs(cycle_time_a - cycle_time_b) / denom

def retrieve_similar_train_cases_temporal(truncated_test, prefix_len, train_cases, examples_count):
    """Picks training cases by 0.5 * normalized control-flow distance + 0.5 * normalized prefix-cycle-time distance"""
    test_activities = extract_prefix_activities(truncated_test, prefix_len)
    test_cycle_time = prefix_cycle_time_seconds(truncated_test, prefix_len)
    variants = build_train_variants(train_cases, prefix_len)
    scored = []

    for variant, cases in variants.items():
        cf_dist = normalized_edit_distance(test_activities, variant)  # same for every case in this variant

        for case in cases:
            case_cycle_time = prefix_cycle_time_seconds(case, prefix_len)
            ct_dist = normalized_cycle_time_distance(test_cycle_time, case_cycle_time)
            score = 0.5 * cf_dist + 0.5 * ct_dist
            scored.append((score, case))

    scored.sort(key=lambda x: x[0])  # lowest combined distance = most similar

    selected = []
    i = 0
    while i < len(scored) and len(selected) < examples_count:
        dist = scored[i][0]

        tied_cases = []
        while i < len(scored) and scored[i][0] == dist:
            tied_cases.append(scored[i][1])
            i += 1

        random.shuffle(tied_cases)
        remaining = examples_count - len(selected)
        selected.extend(tied_cases[:remaining])

    return [deepcopy(c) for c in selected]

def generate_sets(train_cases, test_cases, examples_count, seed, retrieve_fn, mode_label):
    """Runs the shared test-case preparation, then applies retrieve_fn to pick
    training examples for each prepared test case. Also times each retrieval
    for later analysis."""
    prepared_tests = prepare_test_cases(test_cases, seed)
    random.seed(seed + 1)

    result_sets = []
    per_test_timings = []
    total_start = time.perf_counter()

    for test_record in prepared_tests:
        prefix_len = test_record["prefix_len"]
        case_len = test_record["case_len"]
        truncated_test = test_record["test_case"]

        start = time.perf_counter()
        similar_examples = retrieve_fn(truncated_test, prefix_len, train_cases, examples_count)
        elapsed = time.perf_counter() - start
        per_test_timings.append({"prefix_length": prefix_len, "total_case_length": case_len, "selection_time_seconds": elapsed})

        result_sets.append({
            "examples": similar_examples,
            "test_case": truncated_test,
            "prefix_length": prefix_len,
            "total_case_length": case_len,
            "selection_time_seconds": elapsed,
        })

    total_elapsed = time.perf_counter() - total_start
    timing_summary = {
        "mode": mode_label,
        "n_sets": len(result_sets),
        "total_selection_time_seconds": total_elapsed,
        "average_selection_time_seconds": (total_elapsed / len(result_sets)) if result_sets else 0.0,
        "per_test_case": per_test_timings,
    }

    return result_sets, timing_summary

def generate_random_sets(train_cases, test_cases, examples_count, seed):
    """Baseline mode: training examples drawn randomly, independent of the test prefix."""
    return generate_sets(train_cases, test_cases, examples_count, seed, retrieve_fn=retrieve_random_train_cases, mode_label="random")

def generate_similar_prefix_sets(train_cases, test_cases, examples_count, seed):
    """Training examples are the training cases whose control-flow prefix is most
    similar to the test prefix, by normalized DL/OSA distance."""
    return generate_sets(train_cases, test_cases, examples_count, seed, retrieve_fn=retrieve_similar_train_cases, mode_label="similar_prefix")

def generate_similar_prefix_temporal_sets(train_cases, test_cases, examples_count, seed):
    """Training examples are chosen by 0.5 * normalized control-flow distance
    + 0.5 * normalized prefix-cycle-time distance."""
    return generate_sets(train_cases, test_cases, examples_count, seed, retrieve_fn=retrieve_similar_train_cases_temporal, mode_label="similar_prefix_temporal")