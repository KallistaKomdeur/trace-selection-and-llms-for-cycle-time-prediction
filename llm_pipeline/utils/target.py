"""
What the LLM is asked for, and how its answer becomes the quantity the paper scores.

Every row of the comparison is scored on the TOTAL cycle time of the case (R^2 and RA
against actual_case_duration). Two ways of obtaining a total-time prediction from an
LLM coexist:

  * "total_direct"            the prompt asks for the total time (prompts/single_ref.txt);
                              the answer is the prediction.
  * "remaining_plus_elapsed"  the prompt asks for the REMAINING time from the last
                              observed event (prompts/*_remaining.txt); the prediction
                              is that answer plus the elapsed time at the prefix, which
                              is known exactly for a running case.

The configuration name decides: anything ending in "_remaining" is the second kind.
Records keep both the raw answer and the scored prediction, so neither is lost.
"""

REMAINING_SUFFIX = "_remaining"


def answer_mode(configuration):
    return "remaining_plus_elapsed" if str(configuration).endswith(REMAINING_SUFFIX) else "total_direct"


def to_total_time(configuration, llm_answer, elapsed_at_prefix):
    """The total-time prediction scored by the metrics, whatever the prompt asked for."""
    if llm_answer is None:
        return None
    if answer_mode(configuration) == "remaining_plus_elapsed":
        return float(llm_answer) + float(elapsed_at_prefix)
    return float(llm_answer)
