SPLIT_MARKER = {"PART": "===PART:"}

def split_prompt_by_markers(prompt):
    """
    Splits a prompt into multiple parts based on markers (if present).
    """

    if SPLIT_MARKER["PART"] not in prompt:
        return [prompt]

    lines = prompt.splitlines()
    parts = []
    current = []

    for line in lines:
        if line.startswith(SPLIT_MARKER["PART"]):
            if current:
                parts.append("\n".join(current).strip())
                current = []
        current.append(line)

    if current:
        parts.append("\n".join(current).strip())

    return parts
