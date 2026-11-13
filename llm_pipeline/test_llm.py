import json
import os
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import timezone, datetime

from utils.general_utils import get_input
from utils.send_query import LLMSession
from utils.llama_backend import ContextOverflowError, get_shared_backend
from utils.zai_backend import get_shared_zai_backend, FatalApiError
from utils.anthropic_backend import get_shared_anthropic_backend, RefusalError, MaxTokensError
from utils.run_ledger import plan, FIXED_REPETITION
from utils.target import answer_mode, to_total_time
from utils.prompt_filler import fill_prompt
from utils.preprocessing import preprocess_log
from utils.llm_parsing import parse_llm_output
from utils.prompt_splitter import split_prompt_by_markers
from utils.load_config import load_config
from utils.zai_backend import load_backend_config as load_config_backend
from utils.clean_log import clean_log
from utils.generate_test_sets import generate_fixed_sets

SEED = 42

BASE_DIR = Path(__file__).resolve().parent

def ensure_preprocessed(log_name, clean_first):
    """
    Ensure the preprocessed JSONL exists for the given log. If it does not already exist, the log is preprocessed.
    """
    log_dir = BASE_DIR / "logs" / log_name

    if clean_first:
        # If needs to be cleaned, define paths and clean
        original_csv_path = log_dir / f"{log_name}.csv"
        clean_csv_path = log_dir / f"{log_name}_clean.csv"
        preprocessed_path = log_dir / f"{log_name}_clean_preprocessed.jsonl"

        if not clean_csv_path.exists():
            print(f"Creating clean version of {log_name}")
            clean_log(original_csv_path, clean_csv_path, log_name)

        if not preprocessed_path.exists():
            print(f"Preprocessed clean file missing, running preprocessing for clean {log_name}.")
            preprocess_log(log_name, clean_version = True)

    else:
        # If doesn't need to be cleaned, define path and preprocessed
        preprocessed_path = log_dir / f"{log_name}_preprocessed.jsonl"

        if not preprocessed_path.exists():
            print(f"Preprocessed file missing, running preprocessing for {log_name}")
            preprocess_log(log_name, clean_version = False)

def get_results_path(log_name):
    """
    Returns the JSONL file where query logs are added.
    """
    # PPM_RESULTS_DIR keeps concurrent campaigns in separate trees. Records are far
    # larger than the atomic-append size, so two processes appending to one file would
    # interleave mid-line and corrupt it.
    root = Path(os.environ.get("PPM_RESULTS_DIR") or (BASE_DIR / "results"))
    results_dir = root / log_name
    results_dir.mkdir(parents=True, exist_ok=True)
    existing_files = [f for f in results_dir.iterdir() if f.is_file() and f.name.startswith("run_") and f.suffix == ".jsonl"]
    run_nums = [int(f.stem.split("_")[-1]) for f in existing_files if f.stem.split("_")[-1].isdigit()]  # Check how many runs have already been
    next_run = max(run_nums, default=0) + 1                                                             # Find number for next run
    return results_dir / f"run_{next_run}.jsonl"

def test_llm(log_name, provider, model, configuration, pilot=None, api_key=None, sample_frac=1.0):
    """
    Runs LLM predictions over the fixed pre-generated sets and logs each query.

    Resumes from what is already in results/ (R8), can be capped for sizing (R9) and
    runs as many requests in flight as the server has slots for (R10).
    """
    clean_first = config.get("clean_first", False)
    n_runs= config.get("n_runs", 1)
    print_only= config.get("print_only", True)
    truncate_training_examples = config.get("truncate_training_examples", False)
    selection_mode = config.get("selection_mode", "random")
    n_sets = config.get("n_sets", 1)
    examples_count = config.get("examples_count", 10)
    n_prefixes = config.get("n_prefixes", 1)

    # Ensure log is preprocessed
    ensure_preprocessed(log_name, clean_first)
    results_path = get_results_path(log_name)

    # Ensure fixed sets exist
    log_dir = BASE_DIR / "logs" / log_name
    fixed_sets_path = log_dir / f"{log_name}_{selection_mode}_fixed_sets.json"

    if not fixed_sets_path.exists():
        generate_fixed_sets(log_name, n_sets, examples_count, n_prefixes, clean_first, seed=SEED)
        fixed_sets_path = log_dir / f"{log_name}_{selection_mode}_fixed_sets.json"

    with open(fixed_sets_path, "r", encoding="utf-8") as f:
        fixed_combinations = json.load(f)

    skipped = []
    stats = []
    write_lock = threading.Lock()
    thread_local = threading.local()

    # Preflight for the local backend: probe once, up front. Without this a server that
    # is simply not running costs 450s of backoff per case before anything is logged,
    # and it also front-loads the GGUF hashing so R7 metadata is ready before the loop.
    backend = None
    if provider == "llamacpp":
        try:
            backend = get_shared_backend(model if model in ("primary", "secondary") else None)
            backend.probe()
        except Exception as e:
            raise SystemExit(
                f"llama-server unreachable or unusable: {e}\n"
                f"Start it with the flags recorded in config/backend.yaml, then retry."
            )
        meta = backend.static_metadata()
        print(f"Backend: {meta['model_name']} {meta['quant']} | "
              f"{meta['context_per_slot']} tok/slot x {meta['total_slots']} slot(s) | "
              f"thinking={meta['thinking_enabled']} | prefix_cache={meta['prefix_cache_regime']}")
        print(f"Sampling: {meta['sampling']}")

    zai = None
    if provider == "zai":
        zai = get_shared_zai_backend(model if model in ("glm-5.3-flash", "glm-5.2") else None, api_key)
        m = zai.static_metadata()
        print(f"Backend: {m['model_name']} via {m['endpoint']}")
        print(f"Sampling: {m['sampling']} | thinking={m['thinking_enabled']} "
              f"(disableable={m['thinking_disableable']})")
        print(f"Pricing: ${m['price_input_per_mtok']}/Mtok in, ${m['price_output_per_mtok']}/Mtok out")

    anth = None
    if provider == "anthropic":
        known = (load_config_backend().get("anthropic", {}) or {}).get("models", {})
        anth = get_shared_anthropic_backend(model if model in known else None, api_key)
        m = anth.static_metadata()
        print(f"Backend: {m['model_name']} ({m['api_model_id']}) via {m['endpoint']}")
        print(f"Sampling: {m['sampling_sent']} | thinking={m['thinking_mode']} | prefix_cache={m['prefix_cache']}")
        print(f"Pricing: ${m['price_input_per_mtok']}/Mtok in, ${m['price_cached_input_per_mtok']}/Mtok cached, "
              f"${m['price_output_per_mtok']}/Mtok out (synchronous rates)")

    # R8: generations already in results/ are skipped, so a killed run resumes instead
    # of redoing everything from set_index 0.
    # PPM_REPETITION selects the repetition of the design (default 0). It is part of the
    # generation id, so repetition 1 of a case is a new generation and the ledger does
    # not skip it; it is recorded in every record and filtered by the scorer.
    repetition = int(os.environ.get("PPM_REPETITION", FIXED_REPETITION))
    id_kwargs = dict(log_name=log_name, configuration=configuration,
                     selection_mode=selection_mode, provider=provider, model=model,
                     repetition=repetition)
    indices = range(len(fixed_combinations))
    if sample_frac < 1.0:
        # same generator, seed and draw as fewshot_pgtnet.py --sample_frac
        import numpy as np
        rng = np.random.default_rng(42)
        keep = set(rng.choice(len(fixed_combinations), size=max(1, round(len(fixed_combinations) * sample_frac)),
                              replace=False).tolist())
        indices = [i for i in indices if i in keep]
        print(f"Sampling {len(indices)}/{len(fixed_combinations)} sets ({sample_frac:.0%}, seed 42)")
    todo, already_done = plan(indices, results_path.parent, **id_kwargs)
    if already_done:
        print(f"Resuming: {already_done} already completed, {len(todo)} remaining")

    # R9: cap the run so the campaign can be sized before committing to it.
    if pilot:
        todo = todo[:pilot]
        print(f"Pilot mode: {len(todo)} generation(s)")

    # If debugging, only print the prompt
    if print_only:
        first_idx = todo[0][0] if todo else 0
        prompt_text = fill_prompt(log_name=log_name, configuration=configuration,
                                  set_index=first_idx, clean_first=clean_first)[0]
        for part in (split_prompt_by_markers(prompt_text) if "split" in configuration else [prompt_text]):
            print("Split here\n")
            print(part)
        return results_path

    if not todo:
        print("Nothing to do: every generation for this condition is already present.")
        return results_path

    # R10: the server advertises its slots; the client adapts instead of assuming.
    if backend is not None:
        workers = backend.concurrency()
    elif zai is not None:
        workers = int((load_config_backend().get("zai", {}) or {}).get("concurrency", 1))
    elif anth is not None:
        workers = anth.concurrency
    else:
        workers = 1
    print(f"Running {len(todo)} generation(s) with {workers} concurrent worker(s) (repetition {repetition})")

    def get_session():
        """One session per worker thread: LLMSession keeps per-call state."""
        if not hasattr(thread_local, "session"):
            thread_local.session = LLMSession()
        return thread_local.session

    def run_one(item):
        run_idx, gid = item
        session = get_session()
        session.reset()         # Clear cache to prevent information flow

        (prompt_text, true_total_time, prefix_length, include_case_attr,
         include_log_info, inter_case_attr, true_total_length,
         elapsed_at_prefix, true_remaining_time) = fill_prompt(
            log_name=log_name, configuration=configuration,
            set_index=run_idx, clean_first=clean_first)
        prompt_parts = (split_prompt_by_markers(prompt_text) if "split" in configuration else [prompt_text])

        started = time.perf_counter()
        try:
            llm_outputs = session.send_with_retry(session, provider, model, prompt_parts, run_idx)
        except FatalApiError:
            # Affects every case, not this one: let it abort the whole run.
            raise
        except ContextOverflowError as e:
            # R3: the prompt does not fit in the slot. Skip loudly instead of letting
            # llama.cpp truncate it silently, and never retry.
            return None, {"set_index": run_idx, "reason": "context_overflow", "detail": str(e)}, None
        except RefusalError as e:
            # A safety decline (Anthropic, HTTP 200 + stop_reason "refusal"). Counted and
            # reported, never retried and never re-routed to another model.
            return None, {"set_index": run_idx, "reason": "refusal", "detail": str(e)}, None
        except MaxTokensError as e:
            # The model reasoned in the text and ran out of output budget before the
            # answer markers. Skipped, not retried: the budget is fixed by design.
            return None, {"set_index": run_idx, "reason": "max_tokens", "detail": str(e)}, None

        if llm_outputs is None:
            return None, {"set_index": run_idx, "reason": "send_failed"}, None

        try:
            answer = parse_llm_output(llm_outputs[-1])
        except ValueError as e:
            return None, {"set_index": run_idx, "reason": "parse_failed", "detail": str(e)}, None

        if answer is None:
            # parse_llm_output returns None rather than raising when the regex misses.
            # Left unrecorded this silently shrinks n without anyone noticing.
            return None, {"set_index": run_idx, "reason": "unparsable_answer",
                          "detail": (llm_outputs[-1] or "")[:200]}, None

        backend_meta = session.last_metadata()
        record = {
            "generation_id": gid,
            "repetition": repetition,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "provider": provider,
            "model": model,
            "configuration": configuration,
            "log_name": log_name,
            "set_index": run_idx,
            # llm_answer is always the TOTAL-time prediction the metrics score. For a
            # *_remaining configuration it is the model's remaining-time answer plus the
            # elapsed time at the prefix; the raw answer is kept alongside.
            "llm_answer": to_total_time(configuration, answer, elapsed_at_prefix),
            "llm_raw_answer": answer,
            "answer_mode": answer_mode(configuration),
            "target": "total_time",                    # discriminates these records from remaining-time ones
            "actual_case_duration": true_total_time,
            "elapsed_at_prefix": elapsed_at_prefix,    # reference only: remaining = total - elapsed
            "actual_remaining_time": true_remaining_time,

            "prefix_length": prefix_length,
            "true_total_length": true_total_length,
            "truncate_training_examples": truncate_training_examples,
            "case_attributes_included": include_case_attr,
            "log_info_included": include_log_info,
            "included_inter-case_attributes": inter_case_attr,
            "clean_first": clean_first,
            "selection_mode": selection_mode,

            "prompt_parts": prompt_parts,
            "llm_raw_output": llm_outputs,

            # R7: model, quant, GGUF hash, llama.cpp build, server flags, sampling,
            # cache regime and per-call timings. One entry per prompt part.
            "backend_metadata": backend_meta,
        }
        stat = {"wall_s": time.perf_counter() - started, "meta": backend_meta}
        return record, None, stat

    campaign_started = time.perf_counter()
    done = 0
    # Results are consumed as they arrive, not collected first: writing only at the end
    # would leave a whole log's work unrecoverable if the process died, which is exactly
    # what the resume ledger exists to prevent.
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for record, skip, stat in pool.map(run_one, todo):
                done += 1
                if skip is not None:
                    skipped.append(skip)
                    print(f"[{done}/{len(todo)}] skipped ({skip['reason']}) set {skip['set_index']}", flush=True)
                    continue
                stats.append(stat)
                with write_lock:
                    with open(results_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"[{done}/{len(todo)}] set {record['set_index']} -> {record['llm_answer']}", flush=True)
    except FatalApiError as e:
        raise SystemExit(f"\nRun aborted, this cannot be fixed by retrying:\n  {e}")

    campaign_wall = time.perf_counter() - campaign_started

    # Skipped cases used to be collected and then silently discarded, which hid how
    # many generations a reported metric was actually based on.
    if skipped:
        reasons = {}
        for s in skipped:
            reasons[s["reason"]] = reasons.get(s["reason"], 0) + 1
        print(f"\nSkipped {len(skipped)}/{len(todo)}: "
              + ", ".join(f"{k}={v}" for k, v in sorted(reasons.items())))
        with open(results_path.with_suffix(".skipped.json"), "w", encoding="utf-8") as f:
            json.dump(skipped, f, indent=2)

    if stats:
        report_throughput(stats, campaign_wall, workers)

    # Every remote call costs money, so the run closes with what it actually spent.
    remote = zai or anth
    if remote is not None:
        sp = remote.spend()
        print(f"  cost                  : ${sp['usd']:.4f} over {sp['requests']} request(s)")
        print(f"  tokens                : {sp['input_tokens']:,} in / {sp['output_tokens']:,} out")
        if sp['input_tokens'] and sp['usd_without_cache']:
            frac = 100.0 * sp['cached_input_tokens'] / sp['input_tokens']
            saved = sp['usd_without_cache'] - sp['usd']
            print(f"  cache                 : {sp['cached_input_tokens']:,} tok "
                  f"({frac:.1f}% of input), saved ${saved:.4f} "
                  f"({100 * saved / sp['usd_without_cache']:.1f}%)")
        if sp['requests']:
            per = sp['usd']/sp['requests']
            print(f"  per generation        : ${per:.6f}  ->  14110 gen = ${per*14110:.2f}, "
                  f"14526 gen (con BPI11) = ${per*14526:.2f}")
        print("=" * 60)

    return results_path

def report_throughput(stats, wall_s, workers):
    """R9: the numbers needed to size the full campaign before launching it."""
    per_gen = sorted(s["wall_s"] for s in stats)
    p_tok = p_ms = g_tok = g_ms = 0
    for s in stats:
        for m in s["meta"]:
            p_tok += m.get("prompt_tokens") or 0
            p_ms += m.get("prompt_ms") or 0
            g_tok += m.get("completion_tokens") or 0
            g_ms += m.get("predicted_ms") or 0

    print("\n" + "=" * 60)
    print(f"  generations           : {len(stats)}")
    print(f"  wall clock            : {wall_s:.1f} s ({wall_s/60:.1f} min)")
    print(f"  seconds/generation    : mean {statistics.mean(per_gen):.2f}  median {statistics.median(per_gen):.2f}")
    print(f"  throughput            : {len(stats)/wall_s:.2f} gen/s with {workers} worker(s)")
    if p_ms:
        print(f"  prompt processing     : {p_tok/(p_ms/1000):.1f} tok/s  ({p_tok} tok)")
    if g_ms:
        print(f"  generation            : {g_tok/(g_ms/1000):.1f} tok/s  ({g_tok} tok)")
    print("=" * 60)

if __name__ == "__main__":
    log_name, provider, model, configuration, pilot, api_key, sample_frac = get_input()
    config = load_config()

    test_llm(log_name=log_name, provider=provider, model=model,
             configuration=configuration, pilot=pilot, api_key=api_key, sample_frac=sample_frac)
