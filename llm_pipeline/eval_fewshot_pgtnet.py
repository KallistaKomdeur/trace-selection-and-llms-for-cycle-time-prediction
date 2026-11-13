"""Score the few-shot PGTNet campaign: one training per test case, on its 10 examples.

Same metrics and same outlier rule as every other predictor in the paper, so the rows
drop straight into the comparison tables.
"""
import json, sys
from pathlib import Path

import numpy as np, pandas as pd

sys.path.insert(0, "/path/to/LLM_PPM")
from utils.metrics import score as score_pred

BASE = Path("/path/to/PGTNet/experiments_fts5050/fewshot")


def set_base(p):
    """Point the scorer at another campaign (e.g. the control-flow one)."""
    global BASE
    BASE = Path(p)


def score(log, quiet=False):
    rows = [json.loads(l) for l in open(BASE / log / "predictions.jsonl")]
    ok = [r for r in rows if r.get("status") == "ok"]
    if not ok:
        raise SystemExit(f"{log}: no successful run")
    df = pd.DataFrame(ok).drop_duplicates("set_index")
    # Remaining-time campaigns store the model's remaining-time prediction in
    # y_pred_days and the scored total (prediction + elapsed) in y_pred_total_days;
    # y_true_days is the total cycle time in both regimes, so the metrics compare.
    pred_col = "y_pred_total_days" if "y_pred_total_days" in df.columns else "y_pred_days"
    pred = np.clip(df[pred_col].values, 0, None)
    out = score_pred(df["y_true_days"].values, pred)
    keep = pred <= 3 * df["y_true_days"].values.max()
    a = df["y_true_days"].values[keep]
    out["mae_h"] = out.pop("mae") * 24
    out.update(log=log, n_rows=len(rows),
               n_unusable=sum(r.get("status") == "unusable" for r in rows),
               n_error=sum(r.get("status") == "error" for r in rows),
               mean_epochs=float(df["epochs"].mean()),
               mean_train_graphs=float(df["n_train_graphs"].mean()))
    # Trivial 10-shot floors: what you get from the example durations alone, on the
    # same rows and after the same outlier rule. PGTNet has to beat these to be worth
    # training at all.
    for tag, col in (("mean", "ex_mean_days"), ("median", "ex_median_days")):
        if col in df:
            base = np.asarray(df[col][keep])
            bs = score_pred(a, base)
            out[f"baseline_{tag}_r2"] = bs["r2"]
            out[f"baseline_{tag}_mae_h"] = bs["mae"] * 24
            out[f"baseline_{tag}_ra"] = bs["ra"]
    if not quiet:
        print(f"  {log:<24} n={out['n']:5d}/{out['n_rows']:<5d}  "
              f"R2={out['r2']:10.3f}  MAE={out['mae_h']:9.1f} h  "
              f"RA={out['ra']:8.3f} [{out['ra_lo']:.3f}, {out['ra_hi']:.3f}]  "
              f"outlier={out['outliers']:4d}  epoche medie={out['mean_epochs']:.0f}  "
              f"grafi/run={out['mean_train_graphs']:.0f}")
        if "baseline_mean_r2" in out:
            print(f"  {'':<24} baseline mean of the 10 examples: "
                  f"R2={out['baseline_mean_r2']:8.3f}  MAE={out['baseline_mean_mae_h']:9.1f} h  "
                  f"RA={out['baseline_mean_ra']:7.3f}")
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0].startswith("--base="):
        set_base(args.pop(0).split("=", 1)[1])
    logs = args or sorted(p.name for p in BASE.iterdir()
                                  if (p / "predictions.jsonl").exists())
    rows = []
    for l in logs:
        try:
            rows.append(score(l))
        except SystemExit as e:
            print(f"  {l:<24} {e}")
    if rows:
        out = BASE / "results_fewshot.csv"
        new = pd.DataFrame(rows)
        # Scoring one log must not drop the others: the queue calls this per log.
        if out.exists():
            old = pd.read_csv(out)
            new = pd.concat([old[~old.log.isin(new.log)], new], ignore_index=True)
        new.sort_values("log").to_csv(out, index=False)
        print(f"\n-> {out}")
