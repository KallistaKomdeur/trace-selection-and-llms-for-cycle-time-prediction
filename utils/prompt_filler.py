import json
from pathlib import Path

from utils.load_config import load_config
from utils.log_schema import load_log_schema
from utils.generate_test_sets import generate_fixed_sets

SEED = 42

def format_case(case, include_case_attr, case_attr_keys):
    """Formats one case (example or test case) as {case_attrs, ActTimeSeq, total_time}."""
    result = get_case_attributes(case, include_case_attr, case_attr_keys)
    result["ActTimeSeq"] = [[act, t] for act, t in case["ActTimeSeq"]]
    result["total_time"] = case["total_time"]
    return result

def get_case_attributes(case, include_case_attr, case_attribute_keys):
    """Adds schema-defined case attributes depending on settings"""
    if not include_case_attr:
        return {}
    return {k: case[k] for k in case_attribute_keys if k in case}

def fill_prompt(log_name, configuration, set_index):
    """Fills the prompt template with values for one fixed (examples, test_case) set"""
    config = load_config()
    examples_count = config.get("examples_count", 10)
    include_case_attr = config.get("include_case_attributes", False)
    include_log_info = config.get("include_log_info", False)
    selection_mode = config.get("selection_mode", "random")

    root = Path(__file__).resolve().parents[1]
    log_dir = root / "logs" / log_name
    prompt_path = root / "prompts" / f"{configuration}.txt"

    fixed_sets_path = log_dir / f"{log_name}_{selection_mode}_fixed_sets.json"
    if not fixed_sets_path.exists():
        generate_fixed_sets(log_name, examples_count, seed=SEED)

    with open(fixed_sets_path, encoding="utf-8") as f:
        all_sets = json.load(f)

    if set_index >= len(all_sets):
        raise IndexError(f"set_index {set_index} out of range (only {len(all_sets)} sets available)")

    current_set = all_sets[set_index]
    example_cases = current_set["examples"]
    test_case = current_set["test_case"]

    # Some error handling for debugging
    if not prompt_path.exists():
        raise FileNotFoundError(prompt_path)

    schema = load_log_schema(log_name) 
    case_attrs = schema.case_attributes
    case_attr_keys = list(case_attrs) if case_attrs else []

    # Optionally describe the case attributes and the process itself to the LLM
    case_attr_expl = ""
    process_context = ""
    if include_log_info:
        if isinstance(case_attrs, dict):
            case_attr_expl = "\n".join(f'- the key "{k}", which value is {v}' for k, v in case_attrs.items())
        else:
            case_attr_expl = "\n".join(f'- the key "{k}"' for k in case_attr_keys)
        process_context = schema.log_description or ""

    example_blocks = [json.dumps(format_case(case, include_case_attr, case_attr_keys)) for case in example_cases]
    examples_str = "\n\n".join(example_blocks)

    # Sample prediction case from test set
    prefix_length = len(test_case["ActTimeSeq"])
    true_total_time = test_case["true_total_time"]
    true_total_length = test_case["true_total_length"]

    formatted_truncated = format_case(test_case, include_case_attr, case_attr_keys)
    test_block = json.dumps(formatted_truncated, separators=(", ", ": "))

    # Fill template
    template = prompt_path.read_text(encoding="utf-8")
    filled_prompt = (template
        .replace("{EXAMPLES}", examples_str)
        .replace("{NEW_CASE}", test_block)
        .replace("{CASE_ATTRIBUTE_EXPLANATIONS}", case_attr_expl)
        .replace("{PROCESS_CONTEXT}", process_context)
    )

    return filled_prompt, true_total_time, prefix_length, include_case_attr, include_log_info, true_total_length
