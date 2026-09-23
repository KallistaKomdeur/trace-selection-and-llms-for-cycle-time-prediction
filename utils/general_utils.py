import argparse
import json

def get_input():
    """Parses log_name, provider, model and configuration from the command line."""
    parser = argparse.ArgumentParser(description="Run LLM experiment on event log")
    parser.add_argument("log_name", type=str, help="Name of the event log")
    parser.add_argument("provider", type=str, choices=["gemini", "openai", "anthropic"], help="LLM provider")
    parser.add_argument("configuration", type=str, help="Prompt configuration to use (see README)")
    # Optional since Gemini is the provider used in the paper
    # Pass --model to override for OpenAI/Anthropic runs
    parser.add_argument("--model", type=str, default = "2.5-flash", help="Model name for the provider")
    args = parser.parse_args()

    return args.log_name, args.provider, args.model, args.configuration

def load_cases(cases_path): 
    """Loads cases from a preprocessed JSONL file into a list of dicts."""
    cases = [] 
    with open(cases_path, "r", encoding="utf-8") as f:
        for line in f: 
            cases.append(json.loads(line)) 
    return cases