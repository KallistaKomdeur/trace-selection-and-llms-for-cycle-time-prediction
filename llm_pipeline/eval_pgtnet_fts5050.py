"""Score a PGTNet fts5050 run against the fixed sets it was built from.

predictions_matched.csv carries normalised targets (days / max_time_norm, a constant
that is never written to disk). We recover it from real_cycle_time against the truth
in split.json: it has to be identical for every row, so any spread beyond rounding
also proves the (cid, pl) match is the right one and not a coincidental alignment.
"""
import json, sys
from pathlib import Path

import numpy as np, pandas as pd

sys.path.insert(0, "/path/to/LLM_PPM")
from utils.metrics import score as score_pred

EXP = Path("/path/to/PGTNet/experiments_fts5050")


def load_predictions(d):
    """(cid, pl, real_cycle_time, predicted_cycle_time) for every test graph.

    predictions_matched.csv comes from ResultHandler, which skips a dataset whose
    aggregate it "already computed" -- so on a re-used dataset class it hands back the
    previous campaign's file. predictions_raw.csv is this run's inference, in the order
    of the (unshuffled) test loader, i.e. the order of test.pickle; the two are joined by
    position and the join is proven by y: raw real_cycle_time must equal the graph's
    normalised target at every row.
    """
    raw = pd.read_csv(d / "predictions_raw.csv")
    import pickle, torch
    graphs = pickle.load(open(d / "test.pickle", "rb"))
    if len(graphs) != len(raw):
        raise SystemExit(f"{d.name}: {len(raw)} predictions for {len(graphs)} test graphs")
    y_pickle = np.array([float(g.y.reshape(-1)[0]) for g in graphs])
    if not np.allclose(y_pickle, raw["real_cycle_time"].values, rtol=1e-5, atol=1e-7):
        raise SystemExit(f"{d.name}: predictions_raw not aligned with test.pickle")
    return pd.DataFrame({"cid": [str(g.cid) for g in graphs],
                         "pl": [int(g.pl.reshape(-1)[0]) if hasattr(g.pl, "reshape") else int(g.pl)
                                for g in graphs],
                         "real_cycle_time": raw["real_cycle_time"].values,
                         "predicted_cycle_time": raw["predicted_cycle_time"].values})


def score(short, quiet=False):
    d = EXP / short
    df = load_predictions(d)
    split = json.load(open(d / "split.json"))
    spec, test_ids = split["case_spec"], set(split["test"])

    df["cid"] = df["cid"].astype(str)
    df["pl"] = df["pl"].astype(int)
    df = df[df["cid"].isin(test_ids)].drop_duplicates(["cid", "pl"])
    want = {(c, spec[c]["prefix_lengths"][0]) for c in test_ids}
    df = df[[(c, p) in want for c, p in zip(df["cid"], df["pl"])]]
    if df.empty:
        raise SystemExit(f"{short}: no prediction matches a fixed-set test case")

    df["y_true_d"] = [spec[c]["target_days"] for c in df["cid"]]
    # Remaining-time protocol (<short>_full_remaining runs): the graph target is the
    # remaining time after the prefix, so the normalisation constant is recovered from
    # remaining_days, and the scored prediction is predicted remaining + known elapsed,
    # compared with the total time exactly like the LLM rows.
    remaining = "_remaining" in short   # <short>_full_remaining[_seed<N>]
    if remaining:
        df["y_true_rem_d"] = [spec[c]["remaining_days"] for c in df["cid"]]
        df["elapsed_d"] = [spec[c]["elapsed_days"] for c in df["cid"]]
        pos = df["y_true_rem_d"] > 0
        scale = (df.loc[pos, "y_true_rem_d"] / df.loc[pos, "real_cycle_time"])
    else:
        scale = df["y_true_d"] / df["real_cycle_time"]
    rel_spread = float(scale.std() / scale.mean())
    if rel_spread > 1e-6:
        raise SystemExit(f"{short}: normalisation constant is not constant "
                         f"(spread {rel_spread:.2e}) -- the cid/pl match is wrong")
    norm = float(scale.median())
    y_pred = df["predicted_cycle_time"] * norm
    if remaining:
        y_pred = y_pred.clip(lower=0) + df["elapsed_d"]

    out = score_pred(df["y_true_d"].values, y_pred.values)
    out["mae_h"] = out.pop("mae") * 24
    out.update(log=short, n_test=len(test_ids), norm_days=norm)
    if not quiet:
        print(f"  {short:<24} n={out['n']:5d}/{out['n_test']:<5d}  "
              f"R2={out['r2']:9.3f}  MAE={out['mae_h']:9.1f} h  "
              f"RA={out['ra']:7.3f} [{out['ra_lo']:.3f}, {out['ra_hi']:.3f}]  "
              f"outliers={out['outliers']}")
    return out


if __name__ == "__main__":
    shorts = sys.argv[1:] or sorted(p.name for p in EXP.iterdir()
                                    if (p / "predictions_matched.csv").exists())
    rows = []
    for s in shorts:
        try:
            rows.append(score(s))
        except SystemExit as e:
            print(f"  {s:<24} {e}")
    if rows:
        # remaining-time runs get their own CSV; a partial rerun must not drop the other logs
        remaining_rows = [r for r in rows if "_remaining" in r["log"]]
        total_rows = [r for r in rows if "_remaining" not in r["log"]]
        for subset, name in ((total_rows, "results_fts5050.csv"), (remaining_rows, "results_fts5050_remaining.csv")):
            if not subset:
                continue
            out = EXP / name
            df = pd.DataFrame(subset)
            if out.exists():
                old = pd.read_csv(out)
                df = pd.concat([old[~old.log.isin(df.log)], df], ignore_index=True)
            df.sort_values("log").to_csv(out, index=False)
            print(f"\n-> {out}")
