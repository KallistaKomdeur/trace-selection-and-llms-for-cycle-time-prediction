import re

ANSWER_PATTERN = re.compile(r"\[\[\s*##\s*answer\s*##\s*\]\]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)

def parse_llm_output(text):
    """Extracts the numerical answer from the LLM output, or None if not found."""
    if text is None:
        raise ValueError("LLM returned no output")

    match = ANSWER_PATTERN.search(text)
    return float(match.group(1)) if match else None
