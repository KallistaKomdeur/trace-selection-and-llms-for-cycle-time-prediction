#!/usr/bin/env python
"""Turn one similar_prefix_temporal_fixed_sets_5050 JSON into PGTNet inputs.

Everything PGTNet sees comes from the fixed-sets file and nothing else:

  * train/val cases are the *full* cases listed under "examples" (deduplicated);
  * test cases are the truncated prefixes under "test_case", one graph each at
    exactly the fixed set's prefix_length, with the target taken from
    "true_total_time" -- so we never need the full test case, which is not in
    the folder.

Outputs
  new_logs/<short>_fts5050_preprocessed.jsonl   trace-per-line, jsonl_to_xes format
  fts5050_splits/<short>.json                   explicit split + per-case spec

The split file is what GTconvertor consumes instead of train_val_test_ratio:
no case is ever selected by a ratio, the partition is exactly the one the
fixed sets encode.
"""
import argparse, json, os, sys

DEC = json.JSONDecoder()
CHUNK = 1 << 24
CID_KEYS = ("case:concept:name", "Case ID", "case_id", "case", "caseid")


def iter_entries(path):
    """Yield the objects of a top-level JSON array without loading the file.

    ijson rejects the NaN literals these files contain and json.load would need
    ~15x the file size in RAM (the largest is 2.2 GB), so we raw_decode over a
    sliding buffer and drop each entry as soon as it is consumed.
    """
    buf, pos, started = "", 0, False
    with open(path, "r", encoding="utf-8") as f:
        while True:
            if not started:
                while True:
                    i = buf.find("[", pos)
                    if i >= 0:
                        pos, started = i + 1, True
                        break
                    chunk = f.read(CHUNK)
                    if not chunk:
                        return
                    buf, pos = buf[pos:] + chunk, 0
            while True:
                while pos < len(buf) and buf[pos] in " \t\r\n,":
                    pos += 1
                if pos < len(buf):
                    break
                chunk = f.read(CHUNK)
                if not chunk:
                    return
                buf, pos = chunk, 0
            if buf[pos] == "]":
                return
            while True:
                try:
                    obj, end = DEC.raw_decode(buf, pos)
                    break
                except ValueError:
                    chunk = f.read(CHUNK)
                    if not chunk:
                        raise
                    buf, pos = buf[pos:] + chunk, 0
            yield obj
            buf, pos = buf[end:], 0


def build(src, short, val_frac, out_jsonl, out_split):
    train, test, cid_key = {}, {}, None
    n_sets = 0
    dup_conflicts = 0
    for entry in iter_entries(src):
        n_sets += 1
        exs = entry["examples"]
        if cid_key is None and exs:
            cid_key = next(k for k in CID_KEYS if k in exs[0])
        for e in exs:
            cid = str(e[cid_key])
            if cid not in train:
                train[cid] = e
        tc, pl = entry["test_case"], entry["prefix_length"]
        cid = str(tc[cid_key])
        seq = tc["ActTimeSeq"]
        if len(seq) != pl:
            raise SystemExit(f"{short}: case {cid} has {len(seq)} events but prefix_length={pl}")
        prev = test.get(cid)
        if prev is not None:
            # A case reused at a different prefix would need two graphs; the 50/50
            # sets use each test case once, so a clash means our assumption is wrong.
            if prev["pl"] != pl:
                dup_conflicts += 1
            continue
        test[cid] = {"pl": pl, "rec": tc, "target_days": float(tc["true_total_time"]) / 1440.0}
    if dup_conflicts:
        raise SystemExit(f"{short}: {dup_conflicts} test cases appear at more than one prefix length")
    overlap = set(train) & set(test)
    if overlap:
        raise SystemExit(f"{short}: {len(overlap)} case ids are in both train and test")

    # Validation is the temporally last slice of the *example* cases, by case end
    # time. PGTNet needs a validation set for checkpoint selection and this is the
    # only way to get one without touching cases outside the fixed sets.
    order = sorted(train, key=lambda c: (float(train[c]["end_ts"]), c))
    n_val = max(1, int(round(len(order) * val_frac)))
    val_ids = order[len(order) - n_val:]
    train_ids = order[: len(order) - n_val]

    with open(out_jsonl, "w") as f:
        for cid in order:
            rec = dict(train[cid])
            rec[cid_key] = cid
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        for cid, t in test.items():
            rec = dict(t["rec"])
            rec[cid_key] = cid
            rec.pop("true_total_time", None)
            rec.pop("true_total_length", None)
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")

    spec = {}
    for cid in order:
        # Train/val targets come from the fixed sets' own total_time (minutes), not
        # from re-deriving case_end - case_start out of the synthesised timestamps.
        spec[cid] = {"target_days": float(train[cid]["total_time"]) / 1440.0,
                     "prefix_lengths": None}
    for cid, t in test.items():
        spec[cid] = {"target_days": t["target_days"], "prefix_lengths": [t["pl"]]}

    split = {"log": short, "case_id_key": cid_key, "n_sets": n_sets,
             "train": train_ids, "val": val_ids, "test": sorted(test),
             "case_spec": spec}
    with open(out_split, "w") as f:
        json.dump(split, f)

    max_norm = max(spec[c]["target_days"] for c in train_ids + val_ids)
    print(f"{short:38s} sets={n_sets:5d} train={len(train_ids):6d} val={len(val_ids):5d} "
          f"test={len(test):6d} cid_key={cid_key!r} max_time_norm={max_norm:.2f}d")
    return split


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--short", required=True)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--out_split", required=True)
    a = ap.parse_args()
    build(a.src, a.short, a.val_frac, a.out_jsonl, a.out_split)
