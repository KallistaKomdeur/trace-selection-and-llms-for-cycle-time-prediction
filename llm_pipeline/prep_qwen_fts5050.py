"""Install the 50/50 fixed sets where test_llm.py expects them.

The new sets are already the training material; nothing is re-selected here. Two
fields the prompt filler needs are simply missing from them and are derived from
what is there:

    elapsed_at_prefix   = timestamp of the prefix's last event (minutes from start)
    true_remaining_time = true_total_time - elapsed_at_prefix

We also drop a preprocessed JSONL next to them, built from the same file, because
test_llm.py checks that one exists before it will run; it is never read for the
prompts themselves.
"""
import json, os, sys, shutil
from pathlib import Path

sys.path.insert(0, "/path/to/PGTNet/experiments_fts5050")
from build_fts5050 import iter_entries, CID_KEYS

SRC = Path("/path/to/LLM_PPM/fixed_sets_5050")
LOGS = Path("/path/to/LLM_PPM/logs")
MODE = "similar_prefix_temporal"


def prep(src_base, log_name):
    src = SRC / f"{src_base}_similar_prefix_temporal_fixed_sets.json"
    out_dir = LOGS / log_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_sets = out_dir / f"{log_name}_{MODE}_fixed_sets.json"
    out_jsonl = out_dir / f"{log_name}_preprocessed.jsonl"

    # Never clobber a real preprocessed log: for the eight logs already in logs/ that
    # file is the full log and other evaluations read it. It is only written when
    # missing, which is the case for helpdesk, hospital_billing, bpic2012, bpic2015.
    write_jsonl = not out_jsonl.exists()
    seen, n = set(), 0
    tmp = out_sets.with_suffix(".json.part")
    with open(tmp, "w") as fs, open(out_jsonl if write_jsonl else os.devnull, "w") as fj:
        fs.write("[")
        cid_key = None
        for entry in iter_entries(src):
            tc = entry["test_case"]
            if cid_key is None:
                cid_key = next(k for k in CID_KEYS if k in tc)
            seq = tc["ActTimeSeq"]
            elapsed = float(seq[-1][1]) if seq else 0.0
            tc["elapsed_at_prefix"] = elapsed
            tc["true_remaining_time"] = float(tc["true_total_time"]) - elapsed
            fs.write(("," if n else "") + json.dumps(entry, separators=(",", ":")))
            n += 1
            for ex in entry["examples"]:
                cid = str(ex[cid_key])
                if cid not in seen:
                    seen.add(cid)
                    fj.write(json.dumps(ex, separators=(",", ":")) + "\n")
        fs.write("]")
    os.replace(tmp, out_sets)
    size = out_sets.stat().st_size / 1e9
    print(f"{log_name:38s} sets={n:5d} train_cases={len(seen):6d} "
          f"cid_key={cid_key!r} -> {size:.2f} GB "
          f"({'jsonl scritto' if write_jsonl else 'jsonl preesistente, non toccato'})")


if __name__ == "__main__":
    logs = json.load(open("/path/to/PGTNet/experiments_fts5050/logs.json"))
    # PGTNet's short names differ from the LLM side, which keys off the source file
    wanted = sys.argv[1:]
    for l in logs:
        if wanted and l["src"] not in wanted:
            continue
        prep(l["src"], l["src"])
