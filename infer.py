import os
import json
from apiLLM import LLM
from tqdm import tqdm
import time
import re
from concurrent.futures import ThreadPoolExecutor
import argparse

oneshot_examples = {
    "When link": "Here is an example: Question: Given an undirected dynamic graph with the edges [(0, 1, 0), (1, 2, 1), (0, 2, 2)]. When are node 0 and node 2 first connected? Answer:1.",
    "When connect": "Here is an example: Question: Given an undirected dynamic graph with the edges [(0, 1, 0), (1, 2, 1), (0, 2, 2)]. When are node 0 and node 2 first linked? Answer:2.",
    "When triadic closure": "Here is an example: Question: Given an undirected dynamic graph with the edges [(0, 1, 0), (1, 2, 1), (0, 2, 2)]. When are node 2, 0 and 1 first close the triad? Answer:2.",
    "What neighbors at time": "Here is an example: Question: Given an undirected dynamic graph with the edges [(0, 1, 0), (1, 2, 1), (0, 2, 2)].What nodes are linked with node 2 at time 2 ? Answer:[0].",
    "What neighbors in periods": "Here is an example: Question: Given an undirected dynamic graph with the edges [(0, 1, 0), (1, 2, 1), (0, 2, 2)].What nodes are linked with node 2 at or after time 1 ? Answer:[1,2].",
    "Check triadic closure": "Here is an example: Question: Given an undirected dynamic graph with the edges [(0, 1, 0), (1, 2, 1), (0, 2, 2)]. Did node 0, 1 and 2 form a closed triad? Answer:Yes.",
    "Check temporal path": "Here is an example: Question: Given an undirected dynamic graph with the edges [(1, 2, 1), (0, 1, 1), (3, 4, 4)]. Did nodes 0, 1, 2 form a chronological path? Answer: Yes.",
    "Find temporal path": "Here is an example: Question: Given an undirected dynamic graph with the edges [(1, 2, 0), (2, 0, 1), (2, 3, 2)]. Find a chronological path starting from node 1. Answer: [1,2,3].",
    "Sort edge by time": "Here is an example: Question: Given an undirected dynamic graph with the edges [(2, 0, 2), (3, 4, 4), (1, 2, 0), (0, 1, 1)]. Sort the edges by time from earliest to latest. Answer: [(1, 2, 0), (0, 1, 1), (2, 0, 2), (3, 4, 4)]."
}


class InferLLM():

    def __init__(self, api_key, model_name):
        self.model_name = model_name
        self.model_dict = {"Llama3.1": "meta-llama/llama-3.1-8b-instruct",
                           "Llama3.1-70b": "meta-llama/llama-3.1-70b-instruct",
                           "Mistral": "mistralai/mistral-7b-instruct",
                           "Claude3": "anthropic/claude-3-haiku"}
        self.llm = LLM(api_key=api_key, model_name=self.model_dict[model_name])
        self.no_explain_format = " Just output answer without any explanation!"
        self.output_format = self.no_explain_format

    def process_json_files(self, file_path, cot):
        results = []
        with open(file_path, 'r', encoding='utf=8') as infile:
            data = json.load(infile)
        if "EDGE" == cot:
            answers = self.json2llm_edge(data)
            results.extend(answers)
        elif "NO" == cot:
            answers = self.json2llm(data)
            results.extend(answers)
        elif "EXPLAIN" == cot:
            answers = self.json2llm_explain(data)
            results.extend(answers)
        elif "RECONFIRM" == cot:
            answers = self.json2llm_reconfirm(data)
            results.extend(answers)
            results.extend(answers)
        elif "EX_CUR" == cot:
            answers = self.json2llm_ex_cur(data)
            results.extend(answers)
        elif "VERIFY" == cot:
            answers = self.json2llm_ex_verify(data)
            results.extend(answers)
        elif "ALL" == cot:
            answers = self.json2llm_all(data)
            results.extend(answers)
        elif "EX_EDGE" == cot:
            answers = self.json2llm_ex_edge(data)
            results.extend(answers)
        elif "STEP" == cot:
            answers = self.json2llm_think_step(data)
            results.extend(answers)
            results.extend(answers)
        elif "V1" == cot:
            answers = self.json2llm_v1(data)
            results.extend(answers)
        elif "V2" == cot:
            answers = self.json2llm_v2(data)
            results.extend(answers)
        elif "BUILD" == cot:
            answers = self.json2llm_baseline_build(data)
            results.extend(answers)
        elif "CONFIDENCE" == cot:
            answers = self.json2llama_confidence(data)
            results.extend(answers)
        elif "ONE_EX" == cot:
            answers = self.json2llm_one_explain(data)
            results.extend(answers)
        else:
            print("The type doesn't exist !")
        return results

    # no
    def json2llm(self, data):
        results = []
        for item in tqdm(data, desc="Processing one json"):
            input_text = item["instruction"] + "\nNext, please answer: " + item[
                "input"] + self.output_format
            zero_response = self.llm(input_text)
            self.llm.clear_history()
            results.append(
                {"task": item["task"], "input": input_text, "truth": item["output"], "zero_shot_output": zero_response})

        return results

    def json2llm_one_explain(self, data):
        results = []

        for item in tqdm(data, desc="Processing one json"):
            example = oneshot_examples[item["task"]]
            input_text = item["instruction"] + example + "Please explain this example. "
            explain = self.llm(input_text)
            input_text = "\nNext, please answer: " + item["input"] + self.output_format
            zero_response = self.llm(input_text)
            self.llm.clear_history()
            results.append(
                {"task": item["task"], "input": input_text, "truth": item["output"], "zero_shot_output": zero_response})

        return results

    def json2llm_ex_edge(self, data):
        cot_edges = "Describe all the edges in the following problem, for example, (0,1,0) means nodes 0 and 1 are linked at time 0.\n"
        results = []
        for item in tqdm(data, desc="Processing one json"):
            # zero_shot+explain
            input_text = item["instruction"] + cot_edges + item[
                "input"]
            describe0 = self.llm(input_text)
            input_text = item["instruction"] + "\n" + item[
                "input"] + "\nPlease explain the reason in detail."
            describe = self.llm(input_text)
            input_text = "Output the answer without any explanation!"
            zero_response = self.llm(input_text)
            self.llm.clear_history()
            results.append(
                {"task": item["task"], "input": input_text, "truth": item["output"], "zero_shot_output": zero_response,
                 "describe": describe, "describe0": describe0})

        return results

    # explain
    def json2llm_explain(self, data):
        results = []
        for item in tqdm(data, desc="Processing one json"):
            # zero_shot+explain
            input_text = item["instruction"] + "\n" + item[
                "input"] + "\nPlease explain the reason in detail."
            describe = self.llm(input_text)
            input_text = "Output the answer without any explanation!"
            zero_response = self.llm(input_text)
            self.llm.clear_history()
            results.append(
                {"task": item["task"], "input": input_text, "truth": item["output"], "zero_shot_output": zero_response,
                 "describe": describe})

        return results

    def json2llm_v1(self, data):
        results = []
        for item in tqdm(data, desc="Processing one json"):
            # zero_shot+explain
            input_text = item["instruction"] + "\n" + item[
                "input"] + "\nThink about node then time."
            describe = self.llm(input_text)
            input_text = "Output the answer without any explanation!"
            zero_response = self.llm(input_text)
            self.llm.clear_history()
            results.append(
                {"task": item["task"], "input": input_text, "truth": item["output"], "zero_shot_output": zero_response,
                 "describe": describe})

        return results

    def json2llm_v2(self, data):
        results = []
        for item in tqdm(data, desc="Processing one json"):
            # zero_shot+explain
            input_text = item["instruction"] + "\n" + item[
                "input"] + "\nThink about time then node."
            describe = self.llm(input_text)
            input_text = "Output the answer without any explanation!"
            zero_response = self.llm(input_text)
            self.llm.clear_history()
            results.append(
                {"task": item["task"], "input": input_text, "truth": item["output"], "zero_shot_output": zero_response,
                 "describe": describe})

        return results

    def json2llm_baseline_build(self, data):
        results = []
        for item in tqdm(data, desc="Processing one json"):
            # zero_shot+explain
            input_text = item["instruction"] + "\n" + item[
                "input"] + "\nLet's construct a graph with the nodes and edges first."
            describe = self.llm(input_text)
            input_text = "Output the answer without any explanation!"
            zero_response = self.llm(input_text)
            self.llm.clear_history()
            results.append(
                {"task": item["task"], "input": input_text, "truth": item["output"], "zero_shot_output": zero_response,
                 "describe": describe})

        return results

    def json2llama_confidence(self, data):
        confidence = " Give your confidence (0% to 100%) after your answer."
        results = []
        for item in tqdm(data, desc="Processing one json"):
            input_text = item["instruction"] + item[
                "input"] + confidence
            describe = self.llm(input_text)
            print(describe)
            input_text = "Output the answer without any explanation!"
            zero_response = self.llm(input_text)
            self.llm.clear_history()
            results.append(
                {"task": item["task"], "input": input_text, "truth": item["output"], "zero_shot_output": zero_response,
                 "describe": describe})

        return results

    def json2llm_think_step(self, data):
        results = []
        for item in tqdm(data, desc="Processing one json"):
            # zero_shot+explain
            input_text = item["instruction"] + "\n" + item[
                "input"] + "\nPlease think step by step."
            describe = self.llm(input_text)
            input_text = "Output the answer without any explanation!"
            zero_response = self.llm(input_text)
            self.llm.clear_history()
            results.append(
                {"task": item["task"], "input": input_text, "truth": item["output"], "zero_shot_output": zero_response,
                 "describe": describe})

        return results

    # expalin+reconfirm
    def json2llm_reconfirm(self, data):
        results = []
        for item in tqdm(data, desc="Processing one json"):
            # zero_shot+explain
            input_text = item["instruction"] + "\n" + item[
                "input"] + "\nPlease explain the reason in detail."
            describe = self.llm(input_text)
            input_text = "Output the answer without any explanation!" + "Ensure that your answer matches your explanation."
            zero_response = self.llm(input_text)
            self.llm.clear_history()
            results.append(
                {"task": item["task"], "input": input_text, "truth": item["output"], "zero_shot_output": zero_response,
                 "describe": describe})

        return results

    def json2llm_ex_cur(self, data):
        no_assumption = "Do not make any assumptions, just focus on the current conditions."
        results = []
        for item in tqdm(data, desc="Processing one json"):
            input_text = item["instruction"] + "\n" + item[
                "input"] + "\nPlease explain the reason in detail." + no_assumption
            describe = self.llm(input_text)
            input_text = "Output the answer without any explanation!"
            zero_response = self.llm(input_text)
            self.llm.clear_history()
            results.append(
                {"task": item["task"], "input": input_text, "truth": item["output"], "zero_shot_output": zero_response,
                 "describe": describe})
        return results

    # explain+check
    def json2llm_ex_verify(self, data):
        check_nt = "Verify that the nodes,time and edges in the problem do indeed exist."
        results = []
        for item in tqdm(data, desc="Processing one json"):
            input_text = item["instruction"] + "\n" + item[
                "input"] + "\nPlease explain the reason in detail." + check_nt
            describe = self.llm(input_text)
            input_text = "Output the answer without any explanation!"
            zero_response = self.llm(input_text)
            self.llm.clear_history()
            results.append(
                {"task": item["task"], "input": input_text, "truth": item["output"], "zero_shot_output": zero_response,
                 "describe": describe})

        return results

    def json2llm_edge(self, data):
        cot_edges = "Describe all the edges in the following problem, for example, (0,1,0) means nodes 0 and 1 are linked at time 0.\n"
        results = []
        for item in tqdm(data, desc="Processing one json"):
            input_text = item["instruction"] + cot_edges + item[
                "input"]
            describe = self.llm(input_text)
            input_text = "Output the answer without any explanation!"
            zero_response = self.llm(input_text)
            self.llm.clear_history()
            results.append(
                {"task": item["task"], "input": input_text, "truth": item["output"], "zero_shot_output": zero_response,
                 "describe": describe})

        return results

    def json2llm_all(self, data):
        cot_edges = "Describe all the edges in the following problem, for example, (0,1,0) means nodes 0 and 1 are linked at time 0.\n"
        check_nt = "Verify that the nodes,time and edges in the problem do indeed exist."
        results = []
        no_assumption = "Do not make any assumptions, just focus on the current conditions."
        for item in tqdm(data, desc="Processing one json"):
            # zero_shot+explain
            input_text = item["instruction"] + cot_edges + item[
                "input"] + "\nPlease explain the reason in detail." + check_nt + no_assumption
            describe = self.llm(input_text)
            input_text = "Output the answer without any explanation!" + "Ensure that your answer matches your explanation."
            zero_response = self.llm(input_text)
            self.llm.clear_history()
            results.append(
                {"task": item["task"], "input": input_text, "truth": item["output"], "zero_shot_output": zero_response,
                 "describe": describe})

        return results

    def process_all_folders(self, base_folder, base_output, cot):
        answer_types = ['no_answer', 'have_answer']
        for answer_type in answer_types:
            file_path = base_folder + '/' + answer_type
            for file_name in tqdm(os.listdir(file_path), desc=f"Processing folders"):
                print(answer_type + ": " + file_name)
                if os.path.isdir(file_path):
                    output_file_path = os.path.join(base_output, answer_type)
                    if not os.path.isdir(output_file_path):
                        os.mkdir(output_file_path)
                    result = self.process_json_files(file_path + '/' + file_name, cot)
                    # result = []
                    with open(output_file_path + '/' + file_name, 'w', encoding='utf-8') as outfile:
                        json.dump(result, outfile, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="prompting")

    parser.add_argument("--COT", default="NO", help="select the prompting")

    parser.add_argument("-m", "--model_name", default="Llama3.1", help="Enter your model name")

    parser.add_argument("--api_key", default=" ", help="Enter your model name")

    args = parser.parse_args()

    # TODO
    model_name = args.model_name
    # TODO
    inferllm = InferLLM(api_key=" ",
                        model_name=args.model_name)

    start_time = time.time()
    # TODO
    graph = "graphs_n5_t10"
    base_folder_path = "filter_text/" + graph + "/ER"
    base_output_path = "output/" + model_name + "/" + graph + "/ER"
    # TODO
    COT = args.COT
    print(COT)
    if COT:
        base_output_path = base_output_path + "/" + COT
    if not os.path.exists(base_output_path):
        os.makedirs(base_output_path)

    inferllm.process_all_folders(base_folder_path, base_output_path, COT)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total processing time: {elapsed_time:.2f} seconds")
