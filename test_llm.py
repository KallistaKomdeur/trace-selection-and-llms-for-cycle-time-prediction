import json
from pathlib import Path
from datetime import timezone, datetime

from utils.general_utils import get_input
from utils.send_query import send_with_retry
from utils.prompt_filler import fill_prompt
from utils.preprocessing import preprocess_log
from utils.llm_parsing import parse_llm_output
from utils.load_config import load_config
from utils.generate_test_sets import generate_fixed_sets

SEED = 42
BASE_DIR = Path(__file__).resolve().parent

def ensure_preprocessed(log_name):
    """Preprocesses the raw event log into JSONL if that hasn't been done yet."""
    log_dir = BASE_DIR / "logs" / log_name
    preprocessed_path = log_dir / f"{log_name}_preprocessed.jsonl"

    if not preprocessed_path.exists():
        print(f"Preprocessed file missing, running preprocessing for {log_name}")
        preprocess_log(log_name)

def get_results_path(log_name):
    """Returns a numbered JSONL path to append this run's query logs to."""
    results_dir = BASE_DIR / "results" / log_name
    results_dir.mkdir(parents=True, exist_ok=True)
    existing_files = [f for f in results_dir.iterdir() if f.is_file() and f.name.startswith("run_") and f.suffix == ".jsonl"]
    run_nums = [int(f.stem.split("_")[-1]) for f in existing_files if f.stem.split("_")[-1].isdigit()]
    next_run = max(run_nums, default=0) + 1
    return results_dir / f"run_{next_run}.jsonl"

def test_llm(log_name, provider, model, configuration, config):
    """Runs LLM predictions over every entry in the fixed pre-generated set and logs each query."""
    print_only = config.get("print_only", True)
    selection_mode = config.get("selection_mode", "random")
    examples_count = config.get("examples_count", 10)

    ensure_preprocessed(log_name)

    log_dir = BASE_DIR / "logs" / log_name
    fixed_sets_path = log_dir / f"{log_name}_{selection_mode}_fixed_sets.json"

    if not fixed_sets_path.exists():
        generate_fixed_sets(log_name, examples_count, seed=SEED)

    with open(fixed_sets_path, "r", encoding="utf-8") as f:
        fixed_combinations = json.load(f)

    if print_only:
        # Debug mode: preview the first filled prompt without querying an LLM
        prompt_text, *_ = fill_prompt(log_name=log_name, configuration=configuration, set_index=0)
        print(prompt_text)
        return None

    results_path = get_results_path(log_name)
    skipped = []

    for run_idx in range(len(fixed_combinations)):
        prompt_text, true_total_time, prefix_length, include_case_attr, include_log_info, true_total_length = fill_prompt(log_name=log_name, configuration=configuration, set_index=run_idx)
        llm_outputs = send_with_retry(provider, model, prompt_text, run_idx)

        if llm_outputs is None:
            skipped.append(run_idx)
            continue

        final_output = llm_outputs[-1]

        try:
            answer = parse_llm_output(final_output)
        except ValueError as e:
            print(f"Run {run_idx + 1} parsing failed: {e}")
            skipped.append(run_idx)
            continue

        # Build log record for later analysis
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "provider": provider,
            "model": model,
            "configuration": configuration,
            "log_name": log_name,
            "set_index": run_idx,
            "llm_answer": answer,
            "actual_case_duration": true_total_time, 

            "prefix_length": prefix_length,
            "true_total_length": true_total_length,
            "case_attributes_included": include_case_attr,
            "log_info_included": include_log_info,
            "selection_mode": selection_mode,

            "prompt_text": prompt_text,
            "llm_raw_output": llm_outputs
        }

        with open(results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"Logged run {run_idx + 1}/{len(fixed_combinations)}")

    if skipped:
        print(f"Skipped {len(skipped)}/{len(fixed_combinations)} runs: {skipped}")

    return results_path

if __name__ == "__main__":
    log_name, provider, model, configuration = get_input()
    config = load_config()
    test_llm(log_name=log_name, provider=provider, model=model, configuration=configuration, config=config)
