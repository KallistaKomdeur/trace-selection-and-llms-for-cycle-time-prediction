import random
import time
from collections import defaultdict
from copy import deepcopy

from utils.general_utils import elapsed_at, remaining_time

def _edit_distance(seq_a, seq_b):
    """Damerau-Levenshtein distance (OSA) between two activity sequences. 
    Insertion, deletion, substitution, and transposition each cost 1, see paper."""
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
            if (i > 1 and j > 1 and seq_a[i - 1] == seq_b[j - 2] and seq_a[i - 2] == seq_b[j - 1]):
                current[j] = min(current[j], two_back[j - 2] + 1)  # transposition
        two_back, one_back, current = one_back, current, two_back

    return one_back[n]

def _normalized_edit_distance(seq_a, seq_b):
    """Raw DL distance divided by the length of the longer of the two sequences."""
    dist = _edit_distance(seq_a, seq_b)
    denom = max(len(seq_a), len(seq_b), 1)
    return dist / denom

def _extract_prefix_activities(case, prefix_len):
    seq = case.get("ActTimeSeq", [])
    return tuple(e[0] if isinstance(e, (list, tuple)) else str(e) for e in seq[:prefix_len])

def _select_test_cases_randomly(test_cases, n_sets, seed):
    """Sample n_sets test cases uniformly at random with a random prefix length in [2, case_len)."""
    random.seed(seed)

    eligible = [c for c in test_cases if len(c.get("ActTimeSeq", [])) >= 3]
    n_sets = min(n_sets, len(eligible))
    sampled = random.sample(eligible, n_sets)

    result = []
    for test_case_full in sampled:
        seq = test_case_full.get("ActTimeSeq", [])
        case_len = len(seq)
        prefix_len = random.randint(2, case_len - 1)

        truncated_test = deepcopy(test_case_full)
        truncated_test["ActTimeSeq"] = seq[:prefix_len]
        truncated_test["true_total_time"] = test_case_full.get("total_time")
        # Kept for reference: the remaining time is total minus what has already elapsed,
        # so a total-time run can still be read as a remaining-time one after the fact.
        truncated_test["elapsed_at_prefix"] = elapsed_at(seq, prefix_len)
        truncated_test["true_remaining_time"] = remaining_time(test_case_full, prefix_len)
        truncated_test["total_time"] = "RUNNING"
        truncated_test["true_total_length"] = case_len

        result.append({"test_case": truncated_test, "prefix_len": prefix_len, "case_len": case_len})

    return result

def _build_train_variants(train_cases, prefix_len):
    """Group training cases by their prefix control-flow variant, so the distance to the test prefix is computed once per variant."""
    variants = defaultdict(list)
    for case in train_cases:
        if len(case.get("ActTimeSeq", [])) < prefix_len:
            continue
        variants[_extract_prefix_activities(case, prefix_len)].append(case)
    return variants

def _retrieve_similar_train_cases(truncated_test, prefix_len, train_cases, examples_count):
    """Pick the training cases whose prefix variant has the smallest normalized DL distance to the test prefix, searching the 
    full training set, ties are broken randomly."""

    test_activities = _extract_prefix_activities(truncated_test, prefix_len)
    variants = _build_train_variants(train_cases, prefix_len)
    scored_variants = sorted(((_normalized_edit_distance(test_activities, variant), variant) for variant in variants), key=lambda x: x[0])

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

def generate_similar_prefix_sets(train_cases, test_cases, n_sets, examples_count, truncate_train, seed):
    """Select (examples, test_case, prefix_length) triples where the training examples
    are the training cases whose control flow prefix is most similar to the test prefix, using normalized Damerau-Levenshtein control-flow distance"""

    print(f"Sampling {n_sets} test cases uniformly")
    best_tests = _select_test_cases_randomly(test_cases, n_sets, seed)
    random.seed(seed + 1)

    result_sets = []
    total_start = time.perf_counter()

    for test_record in best_tests:
        prefix_len = test_record["prefix_len"]
        case_len = test_record["case_len"]
        truncated_test = test_record["test_case"]

        start = time.perf_counter()
        similar_examples = _retrieve_similar_train_cases(truncated_test, prefix_len, train_cases, examples_count)
        elapsed = time.perf_counter() - start
        print(f"Selected {len(similar_examples)} examples for prefix_len={prefix_len} in {elapsed:.4f}s")

        if truncate_train:
            for ex in similar_examples:
                ex["ActTimeSeq"] = ex.get("ActTimeSeq", [])[:prefix_len]

        result_sets.append({
            "examples": similar_examples,
            "test_case": truncated_test,
            "prefix_length": prefix_len,
            "total_case_length": case_len,
            "mode_truncated_training": truncate_train,
            "selection_time_seconds": elapsed,
        })

    print(f"Total selection time: {time.perf_counter() - total_start:.4f}s")
    return result_sets