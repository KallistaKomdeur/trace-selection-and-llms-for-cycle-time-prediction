import json
from pathlib import Path
from utils.general_utils import load_cases
from utils.load_config import load_config
from utils.set_selection import (generate_random_sets, generate_similar_prefix_sets, generate_similar_prefix_temporal_sets)

def temporal_train_test_split(cases, train_ratio=0.8):
    """Splits cases temporally (assuming cases are already sorted by completion time)"""
    if not cases:
        raise ValueError("Empty case list")

    n_train = max(1, int(len(cases) * train_ratio))
    return cases[:n_train], cases[n_train:]

def generate_fixed_sets(log_name, examples_count, seed):
    """Generates and saves the fixed (examples, test_case) sets for one log. """
    config = load_config()
    selection_mode = config.get("selection_mode", "random")

    root = Path(__file__).resolve().parents[1]
    log_dir = root / "logs" / log_name
    preprocessed_path = log_dir / f"{log_name}_preprocessed.jsonl"

    all_cases = load_cases(preprocessed_path)
    train_cases, test_cases = temporal_train_test_split(all_cases)

    if selection_mode == "similar_prefix":
        print("Generating fixed sets (similar_prefix)")
        sets, timing_summary = generate_similar_prefix_sets(train_cases, test_cases, examples_count, seed)
    elif selection_mode == "similar_prefix_temporal":
        print("Generating fixed sets (similar_prefix_temporal)")
        sets, timing_summary = generate_similar_prefix_temporal_sets(train_cases, test_cases, examples_count, seed)
    elif selection_mode == "random":
        print("Generating fixed sets (random)")
        sets, timing_summary = generate_random_sets(train_cases, test_cases, examples_count, seed)
    else:
        print(f'Unknown selection_mode "{selection_mode}"')
        return None
        
    output_path = log_dir / f"{log_name}_{selection_mode}_fixed_sets.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sets, f, ensure_ascii=False, indent=2)

    timing_path = log_dir / f"{log_name}_{selection_mode}_timings.json"
    with open(timing_path, "w", encoding="utf-8") as f:
        json.dump(timing_summary, f, ensure_ascii=False, indent=2)

    return output_path