import json
import threading
from pathlib import Path

from utils.load_config import load_config
from utils.log_schema import load_log_schema
from utils.generate_test_sets import generate_fixed_sets

SEED = 42

def format_case(case, configuration, include_case_attr, included_inter_case, case_attr_keys):
    """
    Formats one full case.
    """
    # Potentially add case attributes
    result = get_case_attributes(case, include_case_attr, case_attr_keys)
    seq = []

    # If single variant, only include the activity and time since case start
    if configuration.startswith("single"):
        for act, t, *_ in case["ActTimeSeq"]:
            seq.append([act, t])

    # Determines which inter-case features are included in the prompt
    elif configuration.startswith("inter-case"):
        allowed = set(included_inter_case)

        for event in case["ActTimeSeq"]:
            act, t = event[:2]
            ef = event[2] if len(event) == 3 else {}
            seq.append([act, t, {k: v for k, v in ef.items() if k in allowed}])

    result["ActTimeSeq"] = seq
    result["total_time"] = case["total_time"]
    return result

def get_case_attributes(case, include_case_attr, case_attribute_keys):
    """
    Adds schema-defined case attributes depending on settings
    """
    if not include_case_attr:
        return {}

    return {k: case[k] for k in case_attribute_keys if k in case}

_FIXED_SETS_CACHE = {}
_FIXED_SETS_LOCK = threading.Lock()

def _load_fixed_sets(path):
    """
    Memoised load of the fixed sets.

    fill_prompt is called once per set_index, and the sets file runs to tens of MB, so
    re-reading and re-parsing it every call made prompt construction quadratic in the
    number of cases. Pure I/O caching: the returned content is identical.
    """
    key = str(path)
    with _FIXED_SETS_LOCK:
        if key not in _FIXED_SETS_CACHE:
            with open(path, encoding="utf-8") as f:
                _FIXED_SETS_CACHE[key] = json.load(f)
        return _FIXED_SETS_CACHE[key]

def fill_prompt(log_name, configuration, set_index, clean_first):
    """
    Fills the prompt template with values. 
    """
    # Get configuration settings
    config = load_config()
    n_sets = config.get("n_sets", 2)       
    examples_count = config.get("examples_count", 2)
    n_prefixes = config.get("n_prefixes", 2)
    include_case_attr = config.get("include_case_attributes", False)
    included_inter_case = config.get("included_inter_case", [])
    include_log_info = config.get("include_log_info", False)
    selection_mode = config.get("selection_mode", "random")

    # Get file locations
    root = Path(__file__).resolve().parents[1]
    log_dir = root / "logs" / log_name
    prompt_path = root / "prompts" / f"{configuration}.txt"

    fixed_sets_path = log_dir / f"{log_name}_{selection_mode}_fixed_sets.json"

    if not fixed_sets_path.exists():
        generate_fixed_sets(log_name, n_sets, examples_count, n_prefixes, clean_first, seed=SEED)
        fixed_sets_path = log_dir / f"{log_name}_{selection_mode}_fixed_sets.json"
    
    all_sets = _load_fixed_sets(fixed_sets_path)
    
    if set_index >= len(all_sets):
        raise IndexError(f"set_index {set_index} out of range (only {len(all_sets)} sets available)")

    # Sets built for remaining-time prediction label the test case with remaining_time
    # instead of total_time; using them here would silently mislabel every prompt.
    if all_sets and all_sets[0].get("test_case", {}).get("total_time") != "RUNNING":
        raise ValueError(f"{fixed_sets_path} was not built for total-time prediction. Delete it so it is regenerated.")

    current_set = all_sets[set_index]
    example_cases = current_set["examples"]
    test_case = current_set["test_case"]

    # Check which preprocessed file to use
    if clean_first:
        preprocessed_path = log_dir / f"{log_name}_clean_preprocessed.jsonl"
    else:
        preprocessed_path = log_dir / f"{log_name}_preprocessed.jsonl"

    # Some error handling for debugging
    if not preprocessed_path.exists():
        raise FileNotFoundError(preprocessed_path)

    if not prompt_path.exists():
        raise FileNotFoundError(prompt_path)

    # Optionally collect case attribute explanations and log description/context
    case_attr_expl = ""
    process_context = ""
    schema = load_log_schema(log_name) 
    case_attrs = schema.case_attributes
    case_attr_keys = list(case_attrs) if case_attrs else []

    if include_log_info:
        case_attr_expl = "\n".join(f"- the key \"{k}\", which value is {v}" for k, v in (case_attrs or {}).items())
        process_context = schema.log_description or ""

    # Sample and format random examples from training set
    example_blocks = []
    
    for case in example_cases:
        formatted = format_case(case, configuration, include_case_attr, included_inter_case, case_attr_keys)
        example_blocks.append(json.dumps(formatted, separators=(", ", ": ")))

    examples_str = "\n\n".join(example_blocks)

    # Sample prediction case from test set
    prefix_length = len(test_case["ActTimeSeq"])
    true_total_time = test_case["true_total_time"]
    elapsed_at_prefix = test_case["elapsed_at_prefix"]
    true_remaining_time = test_case["true_remaining_time"]
    true_total_length = test_case["true_total_length"]

    formatted_truncated = format_case(test_case, configuration, include_case_attr, included_inter_case, case_attr_keys)
    test_block = json.dumps(formatted_truncated, separators=(", ", ": "))

    # Fill template
    template = prompt_path.read_text(encoding="utf-8")
    filled_prompt = (template
        .replace("{EXAMPLES}", examples_str)
        .replace("{NEW_CASE}", test_block)
        .replace("{CASE_ATTRIBUTE_EXPLANATIONS}", case_attr_expl or "")
        .replace("{PROCESS_CONTEXT}", process_context or "")
        # remaining-time prompts state the elapsed time of the running case explicitly;
        # total-time prompts have no such placeholder and are unchanged by this
        .replace("{ELAPSED}", f"{float(elapsed_at_prefix):.1f}")
    )

    return filled_prompt, true_total_time, prefix_length, include_case_attr, include_log_info, included_inter_case, true_total_length, elapsed_at_prefix, true_remaining_time
