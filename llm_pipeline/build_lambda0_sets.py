"""Fixed sets for the temporal-only selection (lambda = 0) of Section 3.1.

Same protocol as the 50/50 (lambda = 0.5) sets, so the rows are comparable cell by cell:
the test cases, their prefix lengths and the retrieval pool (the temporal 80% training
split of the log) are exactly those of logs/<log>/<log>_similar_prefix_temporal_fixed_sets.json;
only the distance changes, d = d_t (elapsed time of the prefix), through the same
`retrieve` of utils.similar_prefix_lambda with lam = 0. Examples are the completed
training traces, untruncated, 10 per test case, ties shuffled with the same seed.

Output: logs/<log>/<log>_lambda0_fixed_sets.json   (selection_mode: lambda0)
"""
import json, random, sys, time
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/path/to/PGTNet/experiments_fts5050")
from build_fts5050 import iter_entries
from utils.general_utils import load_cases
from utils.data_split import temporal_train_test_split
from utils.similar_prefix_lambda import retrieve

LAM = 0.0
SEED = 42
N_EXAMPLES = 10
LOGS = ["gen_baseline", "gen_drifted", "bpic2020_prepaid_travel_costs", "bpic2020_request_for_payment",
        "bpic2020_domestic_declarations", "bpic2020_international_declarations",
        "bpic2020_travel_permit_data", "bpic2011", "bpic2012"]


def build(log):
    src = ROOT / "logs" / log / f"{log}_similar_prefix_temporal_fixed_sets.json"
    out = ROOT / "logs" / log / f"{log}_lambda0_fixed_sets.json"
    train_cases, _ = temporal_train_test_split(load_cases(ROOT / "logs" / log / f"{log}_preprocessed.jsonl"))
    rng = random.Random(SEED + 1)
    t0 = time.time()
    n = 0
    tmp = out.with_suffix(".json.part")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("[")
        for entry in iter_entries(src):
            tc, k = entry["test_case"], entry["prefix_length"]
            assert len(tc["ActTimeSeq"]) == k and tc.get("total_time") == "RUNNING"
            sel = retrieve(tc["ActTimeSeq"], k, train_cases, N_EXAMPLES, LAM, "equal", rng)
            rec = {"examples": [deepcopy(c) for _, c in sel], "test_case": tc,
                   "prefix_length": k, "total_case_length": entry.get("total_case_length", tc.get("true_total_length")),
                   "mode_truncated_training": False, "lambda": LAM,
                   "distances": [d for d, _ in sel]}
            f.write(("," if n else "") + json.dumps(rec, ensure_ascii=False, separators=(",", ":")))
            n += 1
        f.write("]")
    tmp.replace(out)
    print(f"{log:38s} sets={n:5d} train_pool={len(train_cases):6d} -> {out.name} "
          f"({out.stat().st_size/1e6:.0f} MB, {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    for log in (sys.argv[1:] or LOGS):
        build(log)
