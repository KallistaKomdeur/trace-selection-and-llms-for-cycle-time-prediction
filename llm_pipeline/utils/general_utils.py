import argparse
import json

def get_input():
    """
    Parser to get variable values from input command. Some choices are fixed (see README). 
    """
    parser = argparse.ArgumentParser(description="Run LLM experiment on event log")
    # Required
    parser.add_argument("log_name", type=str, help="Name of the event log")
    parser.add_argument("provider", type=str, choices=["llamacpp", "zai", "gemini", "openai", "anthropic"], help="LLM provider")
    parser.add_argument("configuration", type=str, choices=["single_ref","single_ref_remaining","inter-case_ref","single_split_ref","inter-case_split_ref", "single_reasoning", "inter-case_reasoning", "inter-case_self_select_split", "inter-case_explanations"], help="Input configuration to use (see README)")
    parser.add_argument("--model", type=str, default = "2.5-flash", help="Model name for the provider. For llamacpp: a key of models: in config/backend.yaml (primary/secondary); anything else falls back to active_model")
    # R9: cap the run and print sizing statistics instead of doing the full sweep
    parser.add_argument("--pilot", type=int, nargs="?", const=100, default=None,
                        help="limit the run to N generations (default 100 when given without a value) and report timing/throughput")
    # API key for the remote providers. Passing a secret on the command line leaves it in
    # the shell history and in `ps`, so the bare flag asks for it interactively instead,
    # and "-" reads it from stdin for piping. Without the flag it comes from .env.
    parser.add_argument("--sample-frac", type=float, default=1.0,
                        help="run a uniform random fraction of the sets (seed 42, same draw as "
                             "fewshot_pgtnet.py --sample_frac) for logs too long to sweep in full")
    parser.add_argument("--api-key", nargs="?", const="__ASK__", default=None,
                        help="API key: give a value, use '-' to read stdin, or pass the flag alone to be prompted (not echoed). Defaults to the .env entry.")
    args = parser.parse_args()

    return args.log_name, args.provider, args.model, args.configuration, args.pilot, resolve_api_key(args.api_key), args.sample_frac

def resolve_api_key(raw):
    """Turns the --api-key flag into an actual key, or None to fall back to the environment."""
    if raw is None:
        return None
    if raw == "__ASK__":
        import getpass
        return getpass.getpass("API key (input hidden): ").strip() or None
    if raw == "-":
        import sys
        return sys.stdin.readline().strip() or None
    return raw.strip() or None

def load_cases(cases_path): 
    """ Loads desired cases """ 
    cases = [] 
    with open(cases_path, "r", encoding="utf-8") as f:
        for line in f: 
            cases.append(json.loads(line)) 
    return cases

def elapsed_at(seq, k):
    """ Cumulative elapsed minutes at the k-th event (1-indexed) of an ActTimeSeq """
    if not seq:
        return 0.0
    return float(seq[min(k, len(seq)) - 1][1])

def remaining_time(case, k):
    """
    Remaining time (in minutes) of a case observed up to its k-th event, i.e. the K_time function
    of the paper: the completion time of the case minus the end of the k-th event. Cases that were
    already finished at k have nothing left, hence 0.
    """
    total = case.get("total_time")

    if not isinstance(total, (int, float)):
        return None

    return max(0.0, float(total) - elapsed_at(case.get("ActTimeSeq", []), k))
