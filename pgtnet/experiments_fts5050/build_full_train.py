#!/usr/bin/env python
"""PGTNet inputs for the full-training-set regime on one log.

Train/val are every case of the original event log that is not a fixed-set test case
(for bpic2011: 831 of 1039). Test stays exactly the fixed sets' 208 test cases, each
contributing one graph at its own prefix length with the target taken from
true_total_time -- identical to the few-shot campaigns, so the three PGTNet rows are
scored on the same predictions.
"""
import argparse, json, os, sys

sys.path.insert(0, "/home/padela/Scrivania/PGTNet/experiments_fts5050")
from build_fts5050 import iter_entries, CID_KEYS

LLM = "/home/padela/Scrivania/LLM_PPM"


def main(log, short, val_frac, out_jsonl, out_split, sets="temporal"):
    # The test cases are the same (case, prefix length) pairs in both retrieval families,
    # so either file defines the same test set. bpi12w only has the control-flow one.
    src = (f"{LLM}/similar_prefix_temporal_fixed_sets_5050/{log}_similar_prefix_temporal_fixed_sets.json"
           if sets == "temporal" else
           f"{LLM}/logs/{log}/{log}_similar_prefix_fixed_sets.json")
    if not os.path.exists(src):
        raise SystemExit(f"fixed sets non trovati: {src}")
    test, cid_key = {}, None
    for e in iter_entries(src):
        tc, pl = e["test_case"], e["prefix_length"]
        if cid_key is None:
            cid_key = next(k for k in CID_KEYS if k in tc)
        cid = str(tc[cid_key])
        if len(tc["ActTimeSeq"]) != pl:
            raise SystemExit(f"{cid}: {len(tc['ActTimeSeq'])} eventi ma prefix_length={pl}")
        test.setdefault(cid, {"pl": pl, "rec": tc,
                              "target_days": float(tc["true_total_time"]) / 1440.0})

    train = {}
    with open(f"{LLM}/logs/{log}/{log}_preprocessed.jsonl") as f:
        for line in f:
            o = json.loads(line)
            cid = str(o[cid_key])
            if cid not in test:
                train[cid] = o

    # Validation is the temporally last slice of the training portion, by case end time.
    order = sorted(train, key=lambda c: (float(train[c]["end_ts"]), c))
    n_val = max(1, int(round(len(order) * val_frac)))
    val_ids, train_ids = order[len(order) - n_val:], order[: len(order) - n_val]

    with open(out_jsonl, "w") as f:
        for cid in order:
            rec = dict(train[cid]); rec[cid_key] = cid
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        for cid, t in test.items():
            rec = dict(t["rec"]); rec[cid_key] = cid
            rec.pop("true_total_time", None); rec.pop("true_total_length", None)
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")

    spec = {c: {"target_days": float(train[c]["total_time"]) / 1440.0,
                "prefix_lengths": None} for c in order}
    spec.update({c: {"target_days": t["target_days"], "prefix_lengths": [t["pl"]]}
                 for c, t in test.items()})
    with open(out_split, "w") as f:
        json.dump({"log": short, "case_id_key": cid_key, "train": train_ids,
                   "val": val_ids, "test": sorted(test), "case_spec": spec}, f)

    norm = max(spec[c]["target_days"] for c in order)
    print(f"{short:24s} train={len(train_ids)} val={len(val_ids)} test={len(test)} "
          f"cid_key={cid_key!r} max_time_norm={norm:.1f}d")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--short", required=True)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--out_split", required=True)
    ap.add_argument("--sets", choices=("temporal", "controlflow"), default="temporal")
    a = ap.parse_args()
    main(a.log, a.short, a.val_frac, a.out_jsonl, a.out_split, a.sets)
