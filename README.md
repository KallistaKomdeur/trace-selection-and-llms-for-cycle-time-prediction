# Combining Trace Selection and Large Language Models for Process Cycle Time Prediction README. For the supplementary material check the folder "supplementary material"

## Installation

### 1. Make sure Python is installed

This project requires a Python version between 3.9 and 3.11 (I use 3.10). You can check your version by running the following in the terminal:

```bash
python --version
```

If Python is not installed, download it from: https://www.python.org/downloads/.

### 2. Install dependencies

To install the required dependencies, run the following in the terminal:

```bash
pip install -r requirements.txt
```

### 3. Prepare your log

Since logs are too large to store on GitHub, you'll have to upload yours yourself. To do so:

1. Make a folder "logs" in the root directory.
2. In "logs", make a folder with the name of your event log
3. In "logs/your_event_log", place file "your_event_log.csv"
4. In "config/log_schemas", create a file "your_log.yaml". In here, create a schema for your log as shown in the example yaml files.

### 4. Prepare your environment.

In the project root, add a .env file. This file is ignored (see .gitignore), so your keys won't be pushed to github. In here, define (depending on the provider you want to use):

- GEMINI_API_KEY_X (where X is replaced by numbers, so 0, 1, etc. )
- OPENAI_API_KEY
- ANTHROPIC_API_KEY

## Running the project

### Getting LLM responses

To start the main program, run the following in the terminal:

```
python -m test_llm <log_name> <provider> <configuration> <OPTIONAL: --model>
```

The input parameters are:

1. log_name: name of the event log. Can be any of "your_event_log"
2. provider: name of the LLM provider. Restricted to "gemini", "openai", "anthropic"
3. configuration: which input configuration to use. Restricted to the names of available prompts
4. OPTIONAL --model: which model to use from the LLM provider. Default to gemini 2.5-flash, other options gpt-4o-mini

If you want to only see the prompt and not the result, change in "config/settings.yaml" flag "print_only" to True.

### Evaluating LLM responses

To evaluate LLM results for a particular log, configuration, or provider, run the following in the terminal:

```
python -m evaluate <log_name> <selection_mode>
```

### Benchmark evaluation

XGBoost was used as benchmark. To train and tune an XGBoost model on the data (both raw and cleaned), run the following in the terminal:

```
python -m train_xgboost <log_name> <selection_mode>
```

## Useful information

### Mode description

There are currently five supported modes:

1. Single: the activity sequence consists only of the activity and time since start, no inter-case event-level features
2. Single split: the same as single, but the entire prompt text is spread over multiple queries
3. Inter-case: the activity sequence consists of the activity, time since case start, and all inter-case features selected in the settings file
4. Inter-case split: the same as inter-case, but the entire prompt text is spread over multiple queries
5. Inter-case self-select: the same as inter-case, but a part was added where the LLM is instructed to first select which inter-case features it deems useful, and then instructed to only focus on those inter-case features
6. Single reasoning: the same as single, but reasoning is requested and included in the LLM output

### Settings

Settings can be found in config/settings.yaml. You can change the following settings:

1. n_runs: how many runs (= entire prompts) are done for an experiment
2. examples_count: how many examples are included in the prompt text
3. print_only: whether the experiment is actually run (= false), or an example prompt of those settings is printed (= true)
4. clean_first: whether the raw log is used or the log is cleaned before preprocessing
5. include_case_attributes: whether case attributes are included in the prompt
6. include_log_info: whether a context description is included in the prompt
7. included_inter_case: a list of all inter-case features available. The uncommented ones are the ones actually passed to the prompt
