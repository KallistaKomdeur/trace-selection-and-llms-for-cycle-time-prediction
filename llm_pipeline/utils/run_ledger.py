"""
R8 - Resumability.

Long campaigns die: the server is restarted, the machine reboots, a log is interrupted.
Restarting must redo only the missing generations, never the completed ones.

Each generation carries a deterministic id derived from condition + case + repetition
index, so the same cell of the experimental design always maps to the same id across
processes and across runs. On startup the ledger scans every run_*.jsonl already in the
results directory and reports which ids are done.

The repetition index is currently fixed at 0: the design generates each test case once.
It is part of the id anyway, so adding repetitions later does not invalidate ids that
already exist.
"""
import hashlib
import json
from pathlib import Path

FIXED_REPETITION = 0


def generation_id(log_name, configuration, selection_mode, provider, model, set_index,
                  repetition=FIXED_REPETITION):
    """
    Stable identifier for one cell of the design.

    Deliberately excludes sampling parameters: those are recorded in the metadata, and
    folding them in would silently invalidate the whole ledger on any tweak.
    """
    parts = [str(log_name), str(configuration), str(selection_mode),
             str(provider), str(model), str(set_index), str(repetition)]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:32]


def completed_ids(results_dir):
    """
    Ids already present in results_dir. Tolerates truncated final lines, which happen
    when a process is killed mid-write.
    """
    results_dir = Path(results_dir)
    done = set()
    if not results_dir.exists():
        return done

    for path in sorted(results_dir.glob("run_*.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # partial trailing line from an interrupted run
                gid = rec.get("generation_id")
                if gid:
                    done.add(gid)
    return done


def plan(all_indices, results_dir, **id_kwargs):
    """
    Split the work into (to_run, already_done) given what the results directory holds.
    Returns the list of (set_index, generation_id) still to generate.
    """
    done = completed_ids(results_dir)
    todo = []
    skipped = 0
    for idx in all_indices:
        gid = generation_id(set_index=idx, **id_kwargs)
        if gid in done:
            skipped += 1
        else:
            todo.append((idx, gid))
    return todo, skipped
