# README

## Data Preparation

Ensure that the N, P, and T parameters are consistent across the `graph_data/graph_generator.py`, `text_data/text_generator.py`, and`text_data/text_filter.py`. Please update the paths in these files to your own storage paths.

### Step 1: Graph Generator

Run `python graph_data/graph_generator.py` to generate graphs.

### Step 2: Text Generator

Run `python text_data/text_generator.py` to generate tasks  based on graphs from Step 1.

### Step 3: Text filter

Run `python text_data/text_filter.py` to filter data using the text from Step 2 and obtain an equal number of tasks with and without answers.

## Prompting

### Step 1: Implement your own LLM class

Implement your own LLM class similar to the one in `api_LLM.py`  and `api.py`.  Ensure the implementation allows using `llm()` for conversation and `clear_history()` to clear the conversation history.

### Step 2: Infer

Run the command below to use different prompting template.  The COT parameter corresponds to different prompts.

`python infer.py --model_name Llama3.1 --COT NO --api_key your_api_key`

### Step 3: Evaluate

Run the command below to get the accuracy rate.

`python acc.py --model_name Llama3.1 --COT NO`