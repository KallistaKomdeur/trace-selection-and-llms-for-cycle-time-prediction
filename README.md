# Combining Trace Selection and Large Language Models for Process Cycle Time Prediction

Few-shot LLM prediction of total case duration. Given a partial case, the pipeline builds a prompt with some similar, complete example cases and asks an LLM to predict how long the case will take in total. Three strategies for picking those examples are compared: random, only control-flow similarity, and control-flow + temporal similarity.

## Pipeline overview

1. **Preprocess** a raw CSV event log into a JSONL of cases with case attributes, an activity/time sequence, and a total duration.
2. **Generate fixed test sets** by splitting cases temporally into train/test, then sample test cases (truncated to a random prefix) paired with a fixed set of training examples, chosen by one of the three strategies.
3. **Query the LLM** over every (examples, test_case) pair, parse its numerical answer, and log it.
4. **Evaluate** the performance by aggregating all runs for a dataset.

## Installation

### 1. Make sure Python is installed

This project requires a Python version between 3.9 and 3.11. The Python version can be checked by running the following in the terminal:

```bash
python --version
```

If Python is not installed, download it from: https://www.python.org/downloads/.

### 2. Install dependencies

To install the required dependencies, run the following in the terminal:

```bash
pip install -r requirements.txt
```

### 3. Set up environment.

Create a `.env` file in the repo root with whichever provider keys you will use, using the following names:

- GEMINI_API_KEY
- OPENAI_API_KEY
- ANTHROPIC_API_KEY

### 4. Adding a dataset

1. Create a folder `logs` in the root directory
2. In `logs`, make a folder {event_log_name}
3. In `log/event_log_name`, place file `event_log_name.csv`
4. In `config/log_schemas`, create file `event_log_name.yaml`. Create a schema for your log as shown in `tester.yaml`, or use one of the predefined schemas of frequently used process mining datasets.

### 5. Adding a prompt template

You can try out different prompts by adding a new `.txt` file in `prompts/`. Available placeholders include:

- `{EXAMPLES}`: one JSON object per line, one per training example
- `{NEW_CASE}`: the partial test case
- `{CASE_ATTRIBUTE_EXPLANATIONS}`: filled only if `include_log_info: true`
- `{PROCESS_CONTEXT}`: the log's `log_description`, filled only if
  `include_log_info: true`

Ask the model to mark its final answer as:

```
[[##answer]] <number>
```

## Running

### General settings

The following settings are available in `config/settings.yaml`:

- `examples_count`: how many training examples the LLM receives per test case
- `include_case_attributes`: whether inter-case attributes are included in the prompt
- `include_log_info`: whether a general description of the log is included in the prompt
- `print_only`: for debugging. If set to true, an example prompt is printed in the terminal, but no query is sent to an LLM
- `selection_mode`: which selection mechanism should be used to select examples. The following selection modes are available:
  - random: uniform random training examples
  - similar_prefix: training examples with the closest prefix to the test
    prefix, by control-flow (normalized Damerau-Levenshtein) distance alone
  - similar*prefix_temporal: training examples chosen by 0.5 * normalized control-flow distance + 0.5 normalized prefix-cycle-time distance

### Getting LLM responses

To start the main program, run the following in the terminal:

```
python -m test_llm <log_name> <provider> <configuration> <OPTIONAL: --model>
```

The input parameters are:

1. log_name: name of the event log. Can be any of `event_log_name`
2. provider: name of the LLM provider. Restricted to "gemini", "openai", "anthropic"
3. configuration: which input configuration to use. Restricted to the names of available prompts in `prompts/`
4. OPTIONAL --model: which model to use from the LLM provider. Default to gemini 2.5-flash, as used in the research associated with this repo.

### Evaluating LLM responses

To evaluate LLM results for a particular log, run the following in the terminal:

```
python -m evaluate <log_name>
```
