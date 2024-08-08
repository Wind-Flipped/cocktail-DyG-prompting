# README

## Data Preparation

Ensure that the N, P, and T parameters are consistent across the `graph_generator.py`, `text_generator.py`, and`text_filter.py`. Please update the paths in these files to your own storage paths.

### Step 1: Graph Generator

Run `graph_generator.py` to generate graphs.

### Step 2: Text Generator

Run `text_generator.py` to generate tasks  based on graphs from Step 1.

### Step 3: Text filter

Run `text_filter.py` to filter data using the text from Step 2 and obtain an equal number of tasks with and without answers.

## Prompting

### Step 1: Implement your own LLM class

Implement your own LLM class similar to the one in `api_LLM.py`  and `api.py`.  Ensure the implementation allows using `llm()` for conversation and `clear_history()` to clear the conversation history.

### Step 2: Infer

Run `infer.py` to use different prompting template.  The COT parameter corresponds to different prompts.