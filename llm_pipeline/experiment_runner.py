"""
Campaign runner.

Sweeps the event logs for a given provider/model/configuration by rewriting
config/settings.yaml and launching one test_llm process per log.

Note that yaml.safe_dump rewrites settings.yaml sorted and without comments, so the
file on disk is reformatted every time this runs.
"""
import argparse
import os
import subprocess
from pathlib import Path

import yaml

settings_path = Path("config/settings.yaml")

# BPI11 is deliberately absent. With examples_count=10 its prompts have a median of
# ~30k tokens and a maximum of ~89k (measured with check_prefix.py, truncation on),
# against 16384 tokens per slot on the primary model and 8192 on the secondary.
# Half of its cases cannot be served on this hardware, so it is excluded from the
# local campaign and reported as a hardware limitation.
LOGS = [
    "bpic2020_domestic_declarations",
    "bpic2020_international_declarations",
    "bpic2020_prepaid_travel_costs",
    "bpic2020_request_for_payment",
    "bpic2020_travel_permit_data",
    "gen_baseline",
    "gen_drifted",
]

EXCLUDED = {"bpic2011": "prompts exceed the per-slot context on both local models"}


def main():
    ap = argparse.ArgumentParser(description="Run the LLM campaign over all logs")
    ap.add_argument("--provider", default="llamacpp")
    ap.add_argument("--model", default="primary", help="backend.yaml model key, or a cloud model name")
    ap.add_argument("--configuration", default="single_ref")
    ap.add_argument("--selection-modes", nargs="+", default=["similar_prefix", "random"],
                    help="the random baseline is needed for the comparison to mean anything")
    ap.add_argument("--logs", nargs="+", default=None)
    ap.add_argument("--pilot", type=int, default=None, help="cap generations per log (R9)")
    ap.add_argument("--api-key", nargs="?", const="__ASK__", default=None,
                    help="API key: a value, '-' for stdin, or the flag alone to be prompted once for the whole sweep")
    ap.add_argument("--keep-going", action="store_true",
                    help="do not abort the whole sweep when one log fails")
    args = ap.parse_args()

    # Asked once here and handed to every subprocess through the environment, so the key
    # never appears in a command line and is not repeated per log.
    env = os.environ.copy()
    if args.api_key:
        from utils.general_utils import resolve_api_key
        key = resolve_api_key(args.api_key)
        if key:
            env["ZAI_API_KEY"] = key

    logs = args.logs or LOGS
    for name, why in EXCLUDED.items():
        if name in logs:
            print(f"[!] {name} is excluded: {why}")

    for mode in args.selection_modes:
        with open(settings_path, "r", encoding="utf-8") as f:
            settings = yaml.safe_load(f)
        settings["selection_mode"] = mode
        with open(settings_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(settings, f)

        for log in logs:
            cmd = ["python", "-m", "test_llm", log, args.provider, args.configuration,
                   "--model", args.model]
            if args.pilot:
                cmd += ["--pilot", str(args.pilot)]
            print(f"\n=== {mode} | {log} | {args.provider}:{args.model} ===")
            result = subprocess.run(cmd, env=env)
            if result.returncode != 0:
                if args.keep_going:
                    print(f"[!] {log} failed (exit {result.returncode}), continuing")
                else:
                    raise SystemExit(f"{log} failed with exit {result.returncode}")


if __name__ == "__main__":
    main()
