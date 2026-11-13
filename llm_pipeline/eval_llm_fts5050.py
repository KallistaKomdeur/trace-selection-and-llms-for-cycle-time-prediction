"""Score an LLM campaign on the 50/50 similar-prefix fixed sets.

One row per (log, selection_mode) for the requested provider/model. results/<log>/
holds every campaign that ever ran on that log, so provider, model, target and
selection mode are all filtered rather than assumed.

Scoring follows utils.metrics: each test case is generated once, so R^2 is a point
estimate and RA carries a 95% interval built from the per-case errors.
"""
import argparse, json, sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from utils.metrics import score

LOGS = ["bpic2020_prepaid_travel_costs", "gen_drifted", "gen_baseline",
        "bpic2020_request_for_payment", "bpic2020_domestic_declarations",
        "bpic2020_international_declarations", "bpic2020_travel_permit_data",
        "bpic2011", "helpdesk", "hospital_billing", "bpic2015", "bpic2012"]


def load(log, results_dir):
    recs = []
    d = Path(results_dir) if Path(results_dir).is_absolute() else BASE / results_dir
    d = d / log
    if not d.exists():
        return pd.DataFrame()
    for f in sorted(d.glob("run_*.jsonl")):
        for line in open(f, encoding="utf-8"):
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return pd.DataFrame(recs)


def main(provider, model, mode, logs, results_dir, configuration="single_ref", repetition=0):
    rows = []
    for log in logs:
        df = load(log, results_dir)
        if df.empty:
            print(f"  {log:<38} no results"); continue
        # Older records predate the configuration/answer_mode fields: they are single_ref.
        cfg_col = df["configuration"] if "configuration" in df.columns else "single_ref"
        df = df[(df.get("provider") == provider) & (df.get("model") == model)
                & (df.get("target", "total_time") == "total_time")
                & (df.get("selection_mode") == mode)
                & (cfg_col == configuration)
                & (df["repetition"].fillna(0).astype(int) == repetition if "repetition" in df.columns else repetition == 0)]
        if df.empty:
            print(f"  {log:<38} no record for {provider}/{model}/{mode}"); continue
        # A resumed campaign can re-log a set; the ledger keys on this triple.
        df = df.drop_duplicates(["set_index", "selection_mode", "model"])
        df["llm_answer"] = pd.to_numeric(df["llm_answer"], errors="coerce")
        df["actual_case_duration"] = pd.to_numeric(df["actual_case_duration"], errors="coerce")
        unparsed = int(df["llm_answer"].isna().sum())
        df = df.dropna(subset=["llm_answer", "actual_case_duration"])
        if df.empty:
            print(f"  {log:<38} no parsable answer"); continue
        s = score(df["actual_case_duration"].values, df["llm_answer"].values)
        s.update(log=log, provider=provider, model=model, selection_mode=mode,
                 n_records=len(df), unparsed=unparsed)
        s["mae_h"] = s.pop("mae") / 60
        rows.append(s)
        print(f"  {log:<38} n={s['n']:5d}  R2={s['r2']:7.3f}  MAE={s['mae_h']:9.1f} h  "
              f"RA={s['ra']:6.3f} [{s['ra_lo']:.3f}, {s['ra_hi']:.3f}]  "
              f"outlier={s['outliers']:4d}  non_parsati={unparsed}")
    if rows:
        df = pd.DataFrame(rows)[["log", "provider", "model", "selection_mode", "n_records",
                                 "n", "r2", "mae_h", "ra", "ra_se", "ra_ci", "ra_lo",
                                 "ra_hi", "outliers", "unparsed"]]
        # single_ref keeps the historical file name; other prompts get their own CSV
        suffix = "" if configuration == "single_ref" else f"_{configuration}"
        if repetition:
            suffix += f"_rep{repetition}"
        out = BASE / f"results_{provider}_{model}_{mode}{suffix}.csv".replace("/", "-")
        # A campaign's logs can live in several results trees (old runs, new runs); scoring
        # one tree must not drop the rows another tree produced.
        if out.exists():
            old = pd.read_csv(out)
            df = pd.concat([old[~old.log.isin(df.log)], df], ignore_index=True)
        df.sort_values("log").to_csv(out, index=False)
        print(f"\n  media fra {len(df)} log: R2={df.r2.mean():.3f}  RA={df.ra.mean():.3f}")
        print(f"-> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="llamacpp")
    ap.add_argument("--model", default="primary")
    ap.add_argument("--mode", default="similar_prefix_temporal")
    ap.add_argument("--logs", nargs="*", default=LOGS)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--configuration", default="single_ref",
                    help="prompt configuration to score (records of other prompts are ignored)")
    ap.add_argument("--repetition", type=int, default=0, help="repetition index to score (default 0)")
    a = ap.parse_args()
    main(a.provider, a.model, a.mode, a.logs, a.results_dir, a.configuration, a.repetition)
