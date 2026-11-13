"""
Claude campaign through the Message Batches API.

The full comparison is ~19,700 prompts (9 logs x 2 selection strategies) and ~50M
input tokens. Sent one by one that is ~$100 of input alone; the Batches API bills
every token at half price, and the experiment has no latency requirement (each test
case is scored once, nothing depends on the answer arriving quickly), so the bulk of
the campaign goes through it. The per-prediction latency the paper's Table 1 needs is
measured separately on a synchronous sample (test_llm.py --pilot).

The prompt, the fixed sets, the ledger ids and the result records are exactly those
of test_llm.py: fill_prompt builds the text, run_ledger.generation_id names the cell,
and the record written per answer has the same keys (plus execution="batch" and the
batch id), so eval_llm_fts5050.py scores these results unchanged.

Resumable at every level (R8):
  * generations already in the results tree are never resubmitted;
  * submitted batches are recorded in <results_dir>/batches.jsonl with their custom
    ids, so a restart polls them instead of paying for them twice;
  * errored requests are written to <results_dir>/<log>/batch_errors.json; those that
    failed with invalid_request are not resubmitted unless --retry-invalid is given.

Usage:
    python3 run_anthropic_batch.py --mode similar_prefix_temporal --estimate
    python3 run_anthropic_batch.py --mode random --logs gen_baseline --pilot 20
    python3 run_anthropic_batch.py --mode similar_prefix_temporal          # full
    python3 run_anthropic_batch.py --collect                               # poll only
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

CONFIGURATION = "single_ref"
PROVIDER = "anthropic"

# selection_mode -> settings file whose fixed sets it points at (same as the GLM runs)
SETTINGS = {
    "random": BASE_DIR / "config" / "settings_random.yaml",
    "similar_prefix_temporal": BASE_DIR / "config" / "settings_fts5050.yaml",
    "similar_prefix": BASE_DIR / "config" / "settings_similar_prefix.yaml",   # lambda = 1
    "lambda0": BASE_DIR / "config" / "settings_lambda0.yaml",                 # lambda = 0 (temporal only)
}

# The nine logs of the paper, in the order of its tables.
LOGS = ["bpic2012", "bpic2011", "bpic2020_domestic_declarations",
        "bpic2020_international_declarations", "bpic2020_prepaid_travel_costs",
        "bpic2020_request_for_payment", "bpic2020_travel_permit_data",
        "gen_baseline", "gen_drifted"]

OUTPUT_TOKENS_PER_ANSWER = 25   # measured on the pilot: two markers plus one integer


# ----------------------------------------------------------------------------- io

def ledger_path(results_dir):
    return results_dir / "batches.jsonl"


def read_ledger(results_dir):
    p = ledger_path(results_dir)
    entries = []
    if p.exists():
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    # the last line about a batch id wins (status updates are appended, not rewritten)
    latest = {}
    for e in entries:
        latest[e["batch_id"]] = {**latest.get(e["batch_id"], {}), **e}
    return latest


def append_ledger(results_dir, entry):
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(ledger_path(results_dir), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def errors_path(results_dir, log):
    return results_dir / log / "batch_errors.json"


def read_errors(results_dir, log):
    p = errors_path(results_dir, log)
    return json.load(open(p, encoding="utf-8")) if p.exists() else {}


def write_errors(results_dir, log, errs):
    p = errors_path(results_dir, log)
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(errs, open(p, "w", encoding="utf-8"), indent=2)


def next_results_file(results_dir, log):
    d = results_dir / log
    d.mkdir(parents=True, exist_ok=True)
    nums = [int(f.stem.split("_")[-1]) for f in d.glob("run_*.jsonl") if f.stem.split("_")[-1].isdigit()]
    return d / f"run_{max(nums, default=0) + 1}.jsonl"


# ------------------------------------------------------------------------ prompts

class LogPrompts:
    """Renders the prompts of one (log, mode) through the production fill_prompt."""

    def __init__(self, log, mode, model):
        from utils.prompt_filler import fill_prompt, _FIXED_SETS_CACHE
        from utils.run_ledger import generation_id
        from utils.load_config import load_config
        os.environ["PPM_SETTINGS"] = str(SETTINGS[mode])
        cfg = load_config()
        assert cfg.get("selection_mode") == mode, f"{SETTINGS[mode]} does not select {mode}"
        self.log, self.mode, self.model = log, mode, model
        self.clean_first = bool(cfg.get("clean_first", False))
        self.truncate = bool(cfg.get("truncate_training_examples", False))
        self._fill = fill_prompt
        self._cache = _FIXED_SETS_CACHE
        sets_path = BASE_DIR / "logs" / log / f"{log}_{mode}_fixed_sets.json"
        if not sets_path.exists():
            raise SystemExit(f"{sets_path} missing: the fixed sets of this condition were never built")
        # one pass to learn the size without keeping the file twice in memory
        self.n = len(json.load(open(sets_path, encoding="utf-8")))
        self.rep = int(os.environ.get("PPM_REPETITION", 0))
        self.gid = {i: generation_id(log, CONFIGURATION, mode, PROVIDER, model, i, self.rep) for i in range(self.n)}
        self.idx = {g: i for i, g in self.gid.items()}

    def render(self, i):
        """(prompt_text, fields) for set i, fields being what the record needs."""
        (prompt, true_total, prefix_len, inc_attr, inc_info, inter, true_len,
         elapsed, true_rem) = self._fill(self.log, CONFIGURATION, i, self.clean_first)
        return prompt, dict(actual_case_duration=true_total, prefix_length=prefix_len,
                            case_attributes_included=inc_attr, log_info_included=inc_info,
                            included_inter_case=inter, true_total_length=true_len,
                            elapsed_at_prefix=elapsed, actual_remaining_time=true_rem)

    def release(self):
        self._cache.clear()


def make_record(lp, i, prompt, fields, answer, raw, backend_meta, batch_id):
    from utils.run_ledger import FIXED_REPETITION
    from utils.target import answer_mode, to_total_time
    return {
        "generation_id": lp.gid[i],
        "repetition": lp.rep,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": PROVIDER,
        "model": lp.model,
        "configuration": CONFIGURATION,
        "log_name": lp.log,
        "set_index": i,
        "llm_answer": to_total_time(CONFIGURATION, answer, fields["elapsed_at_prefix"]),
        "llm_raw_answer": answer,
        "answer_mode": answer_mode(CONFIGURATION),
        "target": "total_time",
        "actual_case_duration": fields["actual_case_duration"],
        "elapsed_at_prefix": fields["elapsed_at_prefix"],
        "actual_remaining_time": fields["actual_remaining_time"],
        "prefix_length": fields["prefix_length"],
        "true_total_length": fields["true_total_length"],
        "truncate_training_examples": lp.truncate,
        "case_attributes_included": fields["case_attributes_included"],
        "log_info_included": fields["log_info_included"],
        "included_inter-case_attributes": fields["included_inter_case"],
        "clean_first": lp.clean_first,
        "selection_mode": lp.mode,
        "prompt_parts": [prompt],
        "llm_raw_output": [raw],
        "backend_metadata": [backend_meta],
        "execution": "batch",
        "batch_id": batch_id,
    }


# ------------------------------------------------------------------------ estimate

def estimate(backend, logs, modes, calibrate):
    """
    Cost of the batch campaign before anything is submitted. Token counts come from
    the (free) token-counting endpoint on a sample of prompts per log, scaled by
    characters; without a key the 3.2 chars/token heuristic is used.
    """
    m = backend.model_cfg
    disc = float(m.get("batch_discount", 0.5))
    pin, pread, pwrite, pout = (float(m["price_input_per_mtok"]) * disc, float(m["price_cached_input_per_mtok"]) * disc,
                                float(m["price_cache_write_per_mtok"]) * disc, float(m["price_output_per_mtok"]) * disc)
    rows, tot_nocache, tot_cache = [], 0.0, 0.0
    print(f"{'log':<38}{'mode':<25}{'n':>6}{'tok/prompt':>11}{'prefix':>8}{'$ no-cache':>12}{'$ cached':>10}")
    for mode in modes:
        for log in logs:
            lp = LogPrompts(log, mode, backend.model_key)
            chars, prefix_chars, sample = [], None, []
            step = max(1, lp.n // 8)
            for i in range(lp.n):
                p, _ = lp.render(i)
                chars.append(len(p))
                if prefix_chars is None:
                    prefix_chars = p.index(backend.cache_cfg["prefix_marker"])
                if i % step == 0 and len(sample) < 8:
                    sample.append(p)
            lp.release()
            if calibrate:
                ratio = sum(backend.count_tokens(p) for p in sample) / sum(len(p) for p in sample)
                prefix_tok = backend.count_tokens(sample[0][:prefix_chars])
            else:
                ratio = 1.0 / float(backend.guard.get("chars_per_token", 3.2))
                prefix_tok = prefix_chars * ratio
            tok_total = sum(chars) * ratio
            out_tok = lp.n * OUTPUT_TOKENS_PER_ANSWER
            usd_nocache = tok_total / 1e6 * pin + out_tok / 1e6 * pout
            # every prompt reads the header from cache except one write per batch
            usd_cache = ((tok_total - lp.n * prefix_tok) / 1e6 * pin + lp.n * prefix_tok / 1e6 * pread
                         + prefix_tok / 1e6 * pwrite + out_tok / 1e6 * pout)
            tot_nocache += usd_nocache; tot_cache += usd_cache
            rows.append(dict(log=log, mode=mode, n=lp.n, tok_per_prompt=tok_total / lp.n,
                             prefix_tok=prefix_tok, usd_nocache=usd_nocache, usd_cache=usd_cache))
            print(f"{log:<38}{mode:<25}{lp.n:>6}{tok_total / lp.n:>11.0f}{prefix_tok:>8.0f}"
                  f"{usd_nocache:>12.2f}{usd_cache:>10.2f}", flush=True)
    print(f"\nTOTAL at batch rates: ${tot_nocache:.2f} without cache hits, ${tot_cache:.2f} with full "
          f"prefix-cache hits ({'exact token counts' if calibrate else 'chars/3.2 heuristic'})")
    return tot_nocache, rows


# -------------------------------------------------------------------------- submit

def submit(backend, lp, results_dir, pilot, retry_invalid, dry_run):
    """Submit the generations of one (log, mode) that are neither done nor in flight."""
    from utils.run_ledger import plan
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    todo, done = plan(range(lp.n), results_dir / lp.log, log_name=lp.log, configuration=CONFIGURATION,
                      selection_mode=lp.mode, provider=PROVIDER, model=lp.model, repetition=lp.rep)
    in_flight = set()
    for e in read_ledger(results_dir).values():
        if e.get("status") == "submitted" and e.get("log") == lp.log and e.get("mode") == lp.mode:
            in_flight.update(e.get("custom_ids", []))
    errs = read_errors(results_dir, lp.log)
    skip_invalid = {g for g, v in errs.items() if v.get("type") == "invalid_request"} if not retry_invalid else set()
    todo = [(i, g) for i, g in todo if g not in in_flight and g not in skip_invalid]
    if pilot:
        todo = todo[:pilot]
    print(f"[{lp.log} | {lp.mode}] {lp.n} sets: {done} done, {len(in_flight)} in flight, "
          f"{len(skip_invalid)} invalid, {len(todo)} to submit")
    if not todo:
        return []

    max_req = int(backend.batch_cfg.get("max_requests", 20000))
    max_bytes = int(backend.batch_cfg.get("max_bytes", 200_000_000))
    chunks, cur, cur_bytes, overflow = [], [], 0, []
    for i, g in todo:
        prompt, _ = lp.render(i)
        try:
            backend.check_context(prompt)
        except Exception as e:
            overflow.append({"set_index": i, "reason": "context_overflow", "detail": str(e)})
            continue
        req = Request(custom_id=g, params=MessageCreateParamsNonStreaming(**backend.request_params(prompt)))
        size = len(prompt) + 400
        if cur and (len(cur) >= max_req or cur_bytes + size > max_bytes):
            chunks.append(cur); cur, cur_bytes = [], 0
        cur.append(req); cur_bytes += size
    if cur:
        chunks.append(cur)
    if overflow:
        p = results_dir / lp.log / f"batch_{lp.mode}.skipped.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        json.dump(overflow, open(p, "w"), indent=2)
        print(f"  {len(overflow)} prompt(s) exceed the context window, listed in {p.name}")

    ids = []
    for chunk in chunks:
        if dry_run:
            print(f"  dry run: would submit a batch of {len(chunk)} request(s)")
            continue
        b = backend.client.messages.batches.create(requests=chunk)
        append_ledger(results_dir, {"batch_id": b.id, "log": lp.log, "mode": lp.mode, "model": lp.model,
                                    "n_requests": len(chunk), "status": "submitted",
                                    "created_at": datetime.now(timezone.utc).isoformat(),
                                    "custom_ids": [r["custom_id"] for r in chunk]})
        print(f"  submitted batch {b.id} with {len(chunk)} request(s)", flush=True)
        ids.append(b.id)
    return ids


# ------------------------------------------------------------------------- collect

def collect_one(backend, entry, results_dir, lps):
    """Write the records of one ended batch. Returns (n_ok, n_err, usd)."""
    from utils.llm_parsing import parse_llm_output
    from utils.anthropic_backend import RefusalError
    from utils.llama_backend import BackendError

    log, mode = entry["log"], entry["mode"]
    key = (log, mode)
    if key not in lps:
        lps[key] = LogPrompts(log, mode, entry["model"])
    lp = lps[key]
    out = next_results_file(results_dir, log)
    errs = read_errors(results_dir, log)
    skipped = []
    n_ok = n_err = 0
    usd0 = backend.spend()["usd"]
    static = backend.static_metadata()

    with open(out, "a", encoding="utf-8") as f:
        for res in backend.client.messages.batches.results(entry["batch_id"]):
            g = res.custom_id
            i = lp.idx.get(g)
            if i is None:
                errs[g] = {"type": "unknown_custom_id", "batch_id": entry["batch_id"]}
                n_err += 1
                continue
            if res.result.type != "succeeded":
                err = getattr(res.result, "error", None)
                errs[g] = {"type": getattr(err, "type", res.result.type), "set_index": i,
                           "message": str(getattr(err, "message", ""))[:300], "batch_id": entry["batch_id"]}
                n_err += 1
                continue
            msg = res.result.message
            try:
                answer_text, meta = backend.parse_message(msg, batch=True)
            except RefusalError as e:
                skipped.append({"set_index": i, "reason": "refusal", "detail": str(e)}); n_err += 1
                continue
            except BackendError as e:
                skipped.append({"set_index": i, "reason": "max_tokens", "detail": str(e)}); n_err += 1
                continue
            answer = parse_llm_output(answer_text)
            if answer is None:
                skipped.append({"set_index": i, "reason": "unparsable_answer", "detail": answer_text[:200]}); n_err += 1
                continue
            prompt, fields = lp.render(i)
            rec = make_record(lp, i, prompt, fields, answer, answer_text, {**static, **meta}, entry["batch_id"])
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_ok += 1

    if errs:
        write_errors(results_dir, log, errs)
    if skipped:
        p = out.with_suffix(".skipped.json")
        json.dump(skipped, open(p, "w", encoding="utf-8"), indent=2)
    usd = backend.spend()["usd"] - usd0
    append_ledger(results_dir, {"batch_id": entry["batch_id"], "status": "collected", "n_ok": n_ok,
                                "n_err": n_err, "usd": round(usd, 4), "results_file": str(out),
                                "collected_at": datetime.now(timezone.utc).isoformat()})
    print(f"  collected {entry['batch_id']} [{log} | {mode}]: {n_ok} answers, {n_err} failed/skipped, "
          f"${usd:.4f} -> {out.relative_to(BASE_DIR)}", flush=True)
    return n_ok, n_err, usd


def poll(backend, results_dir, poll_s, lps):
    """Wait for every submitted batch of the ledger, collecting each as it ends."""
    total_ok = total_err = 0
    total_usd = 0.0
    while True:
        pending = [e for e in read_ledger(results_dir).values() if e.get("status") == "submitted"]
        if not pending:
            break
        for e in pending:
            b = backend.client.messages.batches.retrieve(e["batch_id"])
            c = b.request_counts
            if b.processing_status == "ended":
                ok, err, usd = collect_one(backend, e, results_dir, lps)
                total_ok += ok; total_err += err; total_usd += usd
            else:
                print(f"  {e['batch_id']} [{e['log']} | {e['mode']}] {b.processing_status}: "
                      f"{c.processing} processing, {c.succeeded} ok, {c.errored} errored", flush=True)
        if any(backend.client.messages.batches.retrieve(e["batch_id"]).processing_status != "ended"
               for e in read_ledger(results_dir).values() if e.get("status") == "submitted"):
            time.sleep(poll_s)
    return total_ok, total_err, total_usd


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Claude campaign through the Message Batches API")
    ap.add_argument("--mode", choices=sorted(SETTINGS), action="append",
                    help="selection strategy; repeat for both (default: both)")
    ap.add_argument("--logs", nargs="+", default=LOGS)
    ap.add_argument("--model", default=None, help="key of anthropic.models in backend.yaml")
    ap.add_argument("--configuration", default="single_ref",
                    help="prompt template (prompts/<name>.txt); a *_remaining one asks for the "
                         "remaining time and the record stores remaining + elapsed as llm_answer")
    ap.add_argument("--results-dir", default="results_sonnet55")
    ap.add_argument("--pilot", type=int, default=None, help="cap submissions per (log, mode)")
    ap.add_argument("--estimate", action="store_true", help="print the cost estimate and exit")
    ap.add_argument("--no-calibrate", action="store_true", help="estimate with the chars/token heuristic, no API call")
    ap.add_argument("--max-usd", type=float, default=60.0,
                    help="refuse to submit when the no-cache estimate exceeds this (batch rates)")
    ap.add_argument("--collect", action="store_true", help="only poll and collect batches already submitted")
    ap.add_argument("--dry-run", action="store_true", help="build the requests, submit nothing")
    ap.add_argument("--retry-invalid", action="store_true")
    ap.add_argument("--api-key", nargs="?", const="__ASK__", default=None)
    a = ap.parse_args()
    global CONFIGURATION
    CONFIGURATION = a.configuration
    if not (BASE_DIR / "prompts" / f"{CONFIGURATION}.txt").exists():
        raise SystemExit(f"prompts/{CONFIGURATION}.txt does not exist")

    from utils.general_utils import resolve_api_key
    from utils.anthropic_backend import get_shared_anthropic_backend
    backend = get_shared_anthropic_backend(a.model, resolve_api_key(a.api_key))
    results_dir = Path(a.results_dir) if Path(a.results_dir).is_absolute() else BASE_DIR / a.results_dir
    modes = a.mode or ["random", "similar_prefix_temporal"]   # the paper's two paid rows
    poll_s = int(backend.batch_cfg.get("poll_s", 60))
    m = backend.static_metadata()
    print(f"Backend: {m['model_name']} ({m['api_model_id']}) | batch discount {m['batch_discount']} | "
          f"thinking={m['thinking_mode']} | effort={backend.sampling.get('effort')} | prefix_cache={m['prefix_cache']}")
    print(f"Results: {results_dir} | configuration: {CONFIGURATION}")

    lps = {}
    if a.collect:
        ok, err, usd = poll(backend, results_dir, poll_s, lps)
        print(f"\ncollected {ok} answers, {err} failed/skipped, ${usd:.2f} at batch rates")
        return

    if not a.pilot:
        est, _ = estimate(backend, a.logs, modes, calibrate=not a.no_calibrate)
        if a.estimate:
            return
        if est > a.max_usd:
            raise SystemExit(f"estimate ${est:.2f} exceeds --max-usd {a.max_usd:.2f}: nothing submitted")

    for mode in modes:
        for log in a.logs:
            lp = LogPrompts(log, mode, backend.model_key)
            lps[(log, mode)] = lp
            submit(backend, lp, results_dir, a.pilot, a.retry_invalid, a.dry_run)
            lp.release()
    if a.dry_run:
        return

    ok, err, usd = poll(backend, results_dir, poll_s, lps)
    sp = backend.spend()
    print(f"\ncampaign: {ok} answers, {err} failed/skipped")
    print(f"cost at batch rates: ${usd:.2f}  (tokens: {sp['input_tokens']:,} in, of which "
          f"{sp['cached_input_tokens']:,} cache reads / {sp['cache_write_tokens']:,} cache writes; "
          f"{sp['output_tokens']:,} out)")


if __name__ == "__main__":
    main()
