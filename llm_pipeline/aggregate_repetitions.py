"""Mean and standard deviation across repetitions, per (log, strategy), for R^2 and RA.

Sources
  GLM / Qwen / Sonnet : results_<provider>_<model>_<mode>_<configuration>[_rep<N>].csv
                         (one row per log; repetition 0 has no suffix)
  PGTNet few-shot     : PGTNet/experiments_fts5050/fewshot[_l0|_cf]_remaining[_seed<N>]/results_fewshot.csv

Every repetition scores the same test cases with the same metrics (utils.metrics.score),
so the spread is the run-to-run variability of the predictor alone. The std is the
sample one (ddof=1); with a single run it is left empty. Output: a CSV per predictor
and the LaTeX cells "mean ± std" for the paper table.

    python3 aggregate_repetitions.py                      # everything found
    python3 aggregate_repetitions.py --configuration single_ref_remaining
"""
import argparse, glob, re, sys
from pathlib import Path

import numpy as np, pandas as pd

BASE = Path(__file__).resolve().parent
PGT = Path("/path/to/PGTNet/experiments_fts5050")
LLM_LOGS = ["bpic2012", "bpic2011", "bpic2020_domestic_declarations", "bpic2020_international_declarations",
            "bpic2020_prepaid_travel_costs", "bpic2020_request_for_payment", "bpic2020_travel_permit_data",
            "gen_baseline", "gen_drifted"]
PGT_SHORT = {"bpic2012": "bpic2012", "bpic2011": "bpic2011", "bpic2020_domestic": "bpic2020_domestic_declarations",
             "bpic2020_international": "bpic2020_international_declarations",
             "bpic2020_prepaid": "bpic2020_prepaid_travel_costs", "bpic2020_rfp": "bpic2020_request_for_payment",
             "bpic2020_permit": "bpic2020_travel_permit_data", "gen_baseline": "gen_baseline", "gen_drifted": "gen_drifted"}
STRATEGY = {"random": "Random", "lambda0": "lambda=0", "similar_prefix_temporal": "lambda=0.5", "similar_prefix": "lambda=1"}
PGT_SETS = {"fewshot": "lambda=0.5", "fewshot_l0": "lambda=0", "fewshot_cf": "lambda=1"}


def llm_runs(provider, model, mode, configuration):
    """{repetition: DataFrame(log -> r2, ra)}"""
    suffix = "" if configuration == "single_ref" else f"_{configuration}"
    runs = {}
    for p in BASE.glob(f"results_{provider}_{model}_{mode}{suffix}*.csv"):
        m = re.fullmatch(rf"results_{re.escape(provider)}_{re.escape(model)}_{re.escape(mode)}{re.escape(suffix)}(?:_rep(\d+))?\.csv", p.name)
        if not m:
            continue
        rep = int(m.group(1) or 0)
        runs[rep] = pd.read_csv(p).set_index("log")[["r2", "ra", "n"]]
    return runs


def pgt_runs(sets, configuration):
    tsuf = "_remaining" if configuration == "single_ref_remaining" else ""
    runs = {}
    for d in PGT.glob(f"{sets}{tsuf}*"):
        m = re.fullmatch(rf"{re.escape(sets)}{re.escape(tsuf)}(?:_seed(\d+))?", d.name)
        if not m or not (d / "results_fewshot.csv").exists():
            continue
        seed = int(m.group(1) or 42)
        df = pd.read_csv(d / "results_fewshot.csv")
        df["log"] = df["log"].map(PGT_SHORT)
        runs[seed] = df.set_index("log")[["r2", "ra", "n"]]
    return runs


def aggregate(runs, logs):
    rows = []
    for log in logs:
        vals = {k: v.loc[log] for k, v in runs.items() if log in v.index}
        if not vals:
            continue
        r2 = np.array([v.r2 for v in vals.values()], float)
        ra = np.array([v.ra for v in vals.values()], float)
        rows.append(dict(log=log, runs=len(vals), n_mean=float(np.mean([v.n for v in vals.values()])),
                         r2_mean=r2.mean(), r2_std=r2.std(ddof=1) if len(r2) > 1 else np.nan,
                         ra_mean=ra.mean(), ra_std=ra.std(ddof=1) if len(ra) > 1 else np.nan))
    return pd.DataFrame(rows)


def cell(m, s, pct=False):
    if np.isnan(m):
        return ""
    f = (lambda x: f"{x * 100:.0f}") if pct else (lambda x: f"{x:.2f}")
    return f"{f(m)}{'\\%' if pct else ''}" + ("" if np.isnan(s) else f" $\\pm$ \\text{{\\tiny {f(s)}}}")


def report(label, strategy, runs, logs, out_rows):
    if not runs:
        return
    agg = aggregate(runs, logs)
    if agg.empty:
        return
    agg.insert(0, "strategy", strategy); agg.insert(0, "predictor", label)
    out_rows.append(agg)
    k = sorted(runs)
    print(f"\n== {label} | {strategy} | {len(k)} run(s): {k}")
    print(agg[["log", "runs", "r2_mean", "r2_std", "ra_mean", "ra_std"]].round(3).to_string(index=False))
    print("  mean over logs: R2 %.3f  RA %.3f" % (agg.r2_mean.mean(), agg.ra_mean.mean()))


def main(configuration):
    out = []
    for label, prov, model in (("GLM-5.3-Flash", "zai", "glm-5.3-flash"), ("Qwen3.5-9B", "llamacpp", "primary"),
                               ("Claude Sonnet 5.5", "anthropic", "claude-sonnet-5-5")):
        for mode, strategy in STRATEGY.items():
            report(label, strategy, llm_runs(prov, model, mode, configuration), LLM_LOGS, out)
    for sets, strategy in PGT_SETS.items():
        report("PGTNet few-shot", strategy, pgt_runs(sets, configuration), LLM_LOGS, out)
    if out:
        df = pd.concat(out, ignore_index=True)
        df["r2_cell"] = [cell(m, s) for m, s in zip(df.r2_mean, df.r2_std)]
        df["ra_cell"] = [cell(m, s, pct=True) for m, s in zip(df.ra_mean, df.ra_std)]
        p = BASE / f"aggregated_{configuration}.csv"
        df.to_csv(p, index=False)
        print(f"\n-> {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--configuration", default="single_ref_remaining")
    main(ap.parse_args().configuration)
