import json
import argparse
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

BASE_DIR = Path(__file__).resolve().parent

def bootstrap_mae_ci(y_true, y_pred, n_boot=1000, ci=95, seed=42):
    """Returns (mae_std, ci_low, ci_high) for MAE via bootstrap resampling of the absolute errors"""
    rng = np.random.default_rng(seed)
    abs_errors = np.abs(np.array(y_true) - np.array(y_pred))
    mae_std = float(np.std(abs_errors, ddof=1))

    boot_means = [np.mean(rng.choice(abs_errors, size=len(abs_errors), replace=True)) for _ in range(n_boot)]
    lo = (100 - ci) / 2
    ci_low, ci_high = np.percentile(boot_means, [lo, 100 - lo])
    return mae_std, float(ci_low), float(ci_high)

def load_results(log_name):
    """Loads and concatenates every results/<log_name>/run_*.jsonl file"""
    results_dir = BASE_DIR / "results" / log_name

    if not results_dir.exists():
        raise FileNotFoundError(f"No results found for log '{log_name}' at {results_dir}")

    all_records = []
    for file_path in results_dir.glob("run_*.jsonl"):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    all_records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    return pd.DataFrame(all_records)

def evaluate(log_name):
    """Evaluates all logged results for a log (R^2, MAE, MAE CI, sample counts)."""
    df = load_results(log_name)

    if df.empty:
        raise ValueError(f"No result records found for log '{log_name}'")

    # Convert to numeric and drop invalid rows
    df['llm_answer'] = pd.to_numeric(df['llm_answer'], errors='coerce')
    df['actual_case_duration'] = pd.to_numeric(df['actual_case_duration'], errors='coerce')
    n_total = len(df)
    df = df.dropna(subset=["llm_answer", "actual_case_duration"])
    n_dropped = n_total - len(df)
    if n_dropped:
        print(f"Dropped {n_dropped}/{n_total} records with missing/unparsable answers")

    group_cols = ['provider', 'model','configuration', 'case_attributes_included', 'log_info_included', 'selection_mode']
    grouped = df.groupby(group_cols)

    summary_results = []

    # Compute metrics for all groups
    for names, group in grouped:
        y_true = group['actual_case_duration']
        y_pred = group['llm_answer']

        group_params = dict(zip(group_cols, names))
        mae_std, ci_low, ci_high = bootstrap_mae_ci(y_true.values, y_pred.values)

        metrics = {
            "parameters": group_params,
            "n_samples": int(len(group)),
            "actual_avg": float(y_true.mean()),
            "predicted_avg": float(y_pred.mean()),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "mae_std": mae_std,
            "mae_ci_low": ci_low,
            "mae_ci_high": ci_high,
            "r2": float(r2_score(y_true, y_pred)),
        }
        summary_results.append(metrics)

    final_output = {"log_name": log_name, "evaluation_groups": summary_results}

    output_path = BASE_DIR / "results" / log_name / "evaluation_summary.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4)

    print(f"Saved evaluation summary to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_name", type=str)
    args = parser.parse_args()
    evaluate(args.log_name)