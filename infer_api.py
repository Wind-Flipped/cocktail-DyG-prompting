import os
import json
# from LLM import LLaMA3_LLM, Qwen2_LLM, Phi3Small_LLM, Gemma2_LLM, ChatGLM4_LLM, Yi_LLM
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
            answers = self.json2llama_edge(data)
            results.extend(answers)
        elif "NO" == cot:
            answers = self.json2llama(data)
            results.extend(answers)
        elif "PATCH" == cot:
            answers = self.json2llama_patch(data)
            results.extend(answers)
        elif "CHECK" == cot:
            answers = self.json2llama_check(data)
            results.extend(answers)
        elif "EXPLAIN" == cot:
            answers = self.json2llama_explain(data)
            results.extend(answers)
        elif "RECONFIRM" == cot:
            answers = self.json2llama_reconfirm(data)
            results.extend(answers)
        elif "ASSUMPTION" == cot:
            answers = self.json2llama_cur(data)
            results.extend(answers)
        elif "EX_CUR" == cot:
            answers = self.json2llama_ex_cur(data)
            results.extend(answers)
        elif "EX_CHECK" == cot:
            answers = self.json2llama_ex_check(data)
            results.extend(answers)
        elif "EX_NOCODE" == cot:
            answers = self.json2llama_ex_nocode(data)
            results.extend(answers)
        elif "EX_ALL" == cot:
            answers = self.json2llama_ex_all(data)
            results.extend(answers)
        elif "ALL" == cot:
            answers = self.json2llama_all(data)
            results.extend(answers)
        elif "CONNECT_LINK" == cot:
            answers = self.json2llama_ex_link_connect(data)
            results.extend(answers)
        elif "EX_EDGE" == cot:
            answers = self.json2llama_ex_edge(data)
            results.extend(answers)
        elif "EXPLAIN1" == cot:
            answers = self.json2llama_explain1(data)
            results.extend(answers)
        elif "EXPLAIN2" == cot:
            answers = self.json2llama_explain2(data)
            results.extend(answers)
        elif "EX_ALL1" == cot:
            answers = self.json2llama_ex_all1(data)
            results.extend(answers)
        else:
            print("The type doesn't exist !")
        return results


    # no
    def json2llama(self, data):
        results = []
        for item in tqdm(data, desc="Processing one json"):
            input_text = item["instruction"] + "\nNext, please answer: " + item[
                "input"] + self.output_format
            zero_response = self.llm(input_text)
            self.llm.clear_history()
            results.append(
                {"task": item["task"], "input": input_text, "truth": item["output"], "zero_shot_output": zero_response})

        return results


    def json2llama_ex_edge(self, data):
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
    def json2llama_explain(self, data):
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


    def json2llama_explain1(self, data):
        results = []
        for item in tqdm(data, desc="Processing one json"):
            # zero_shot+explain
            input_text = item["instruction"] + "\n" + item[
                "input"] + "\nPlease  explain the reasons in detail."
            describe = self.llm(input_text)
            input_text = "Output the answer without any explanation!"
            zero_response = self.llm(input_text)
            self.llm.clear_history()
            results.append(
                {"task": item["task"], "input": input_text, "truth": item["output"], "zero_shot_output": zero_response,
                 "describe": describe})

        return results


    def json2llama_explain2(self, data):
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
    def json2llama_reconfirm(self, data):
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


    def json2llama_ex_link_connect(self, data):
        results = []
        link_connect = "Two nodes are linked if and only if there is a direct edge between them. Two nodes are connected if and only if they can be indirectly connected through other nodes, meaning there exists a path between them."
        for item in tqdm(data, desc="Processing one json"):
            # zero_shot+explain
            input_text = item["instruction"] + "\n" + link_connect + item[
                "input"] + "\nPlease explain the reason in detail."
            describe = self.llm(input_text)
            input_text = "Output the answer without any explanation!" + "Ensure that your answer matches your explanation."
            zero_response = self.llm(input_text)
            self.llm.clear_history()
            results.append(
                {"task": item["task"], "input": input_text, "truth": item["output"], "zero_shot_output": zero_response,
                 "describe": describe})

        return results


    # explain+no_assumption
    def json2llama_ex_cur(self, data):
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
    def json2llama_ex_check(self, data):
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


    # explain+all
    def json2llama_ex_all(self, data):
        check_nt = "Verify that the nodes,time and edges in the problem do indeed exist."
        results = []
        no_assumption = "Do not make any assumptions, just focus on the current conditions."
        for item in tqdm(data, desc="Processing one json"):
            # zero_shot+explain
            input_text = item["instruction"] + "\n" + item[
                "input"] + "\nPlease explain the reason in detail." + check_nt + no_assumption
            describe = self.llm(input_text)
            input_text = "Output the answer without any explanation!" + "Ensure that your answer matches your explanation."
            zero_response = self.llm(input_text)
            self.llm.clear_history()
            results.append(
                {"task": item["task"], "input": input_text, "truth": item["output"], "zero_shot_output": zero_response,
                 "describe": describe})

        return results


    def json2llama_ex_all1(self, data):
        check_nt = "Verify that the nodes,time and edges in the problem do indeed exist."
        results = []
        no_assumption = "Do not make any assumptions, just focus on the current conditions."
        for item in tqdm(data, desc="Processing one json"):
            # zero_shot+explain
            input_text = item["instruction"] + "\n" + item[
                "input"] + "\nPlease explain the reason in detail." + check_nt + no_assumption
            describe = self.llm(input_text)
            input_text = "Output the answer without any explanation!" + "Ensure that your answer matches your explanation."
            zero_response = self.llm(input_text)
            self.llm.clear_history()
            results.append(
                {"task": item["task"], "input": input_text, "truth": item["output"], "zero_shot_output": zero_response,
                 "describe": describe})

        return results


    # def json2llama_ex_all1(data):
    #     check_nt = "Verify that the nodes,time and edges in the problem do indeed exist."
    #     results = []
    #     no_assumption = "Do not make any assumptions, just focus on the current conditions."
    #     for item in tqdm(data, desc="Processing one json"):
    #         # zero_shot+explain
    #         input_text = item["instruction"] + "\n" + item[
    #             "input"] + "\nPlease explain the reason."
    #         describe = llm(input_text)
    #         llm(check_nt)
    #         llm(no_assumption)
    #         input_text = "Give the answer to the next question without explanation." + item[
    #             "instruction"] + "\nNext, please answer: " + item[
    #                          "input"] + " Just output answer without any explanation!" + "Ensure that your answer matches your explanation."
    #         zero_response = llm(input_text)
    #         llm.clear_history()
    #         results.append(
    #             {"task": item["task"], "input": input_text, "truth": item["output"], "zero_shot_output": zero_response,
    #              "describe": describe})
    #
    #     return results


    # check
    def json2llama_check(self, data):
        check_nt = "Verify that the nodes,time and edges in the problem do indeed exist."
        results = []
        for item in tqdm(data, desc="Processing one json"):
            input_text = item["instruction"] + check_nt + item[
                "input"]
            describe = self.llm(input_text)
            input_text = "Output the answer without any explanation!"
            zero_response = self.llm(input_text)
            self.llm.clear_history()
            results.append(
                {"task": item["task"], "input": input_text, "truth": item["output"], "zero_shot_output": zero_response,
                 "describe": describe})

        return results


    def json2llama_ex_nocode(self, data):
        results = []
        nocode = "Use natural language to explain instead of code."
        for item in tqdm(data, desc="Processing one json"):
            # zero_shot+explain
            input_text = item["instruction"] + "\n" + item[
                "input"] + "\nPlease explain the reason in detail." + nocode
            describe = self.llm(input_text)
            input_text = "Output the answer without any explanation!"
            zero_response = self.llm(input_text)
            self.llm.clear_history()
            results.append(
                {"task": item["task"], "input": input_text, "truth": item["output"], "zero_shot_output": zero_response,
                 "describe": describe})

        return results

    def json2llama_cur(self, data):
        no_assumption = "Do not make any assumptions, just focus on the current conditions."
        results = []
        for item in tqdm(data, desc="Processing one json"):
            input_text = item["instruction"] + no_assumption + item[
                "input"]
            describe = self.llm(input_text)
            input_text = "Output the answer without any explanation!"
            zero_response = self.llm(input_text)
            self.llm.clear_history()
            results.append(
                {"task": item["task"], "input": input_text, "truth": item["output"], "zero_shot_output": zero_response,
                 "describe": describe})
        return results


    def json2llama_edge(self, data):
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


    def json2llama_patch(self, data):
        patch4link_forever = "As long as two nodes are linked,they will be linked forever."
        results = []
        for item in tqdm(data, desc="Processing one json"):
            input_text = item["instruction"] + patch4link_forever + item[
                "input"]
            describe = self.llm(input_text)
            input_text = "Output the answer without any explanation!"
            zero_response = self.llm(input_text)
            self.llm.clear_history()
            results.append(
                {"task": item["task"], "input": input_text, "truth": item["output"], "zero_shot_output": zero_response,
                 "describe": describe})

        return results


    def json2llama_all(self, data):
        cot_edges = "Describe all the edges in the following problem, for example, (0,1,0) means nodes 0 and 1 are linked at time 0.\n"
        check_nt = "Verify that the nodes,time and edges in the problem do indeed exist."
        results = []
        no_assumption = "Do not make any assumptions, just focus on the current conditions."
        for item in tqdm(data, desc="Processing one json"):
            # zero_shot+explain
            input_text = item["instruction"] + cot_edges + item[
                "input"]
            describe0 = self.llm(input_text)
            input_text = "\nPlease explain the reason in detail." + check_nt + no_assumption
            describe = self.llm(input_text)
            input_text = "Output the answer without any explanation!" + "Ensure that your answer matches your explanation."
            zero_response = self.llm(input_text)
            self.llm.clear_history()
            results.append(
                {"task": item["task"], "input": input_text, "truth": item["output"], "zero_shot_output": zero_response,
                 "describe": describe, "describe0": describe0})

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


def main():
    # parser = argparse.ArgumentParser(description="myPaper")
    #
    # parser.add_argument('--func', type=str, required=True)
    # parser.add_argument('--cuda_device', type=str, required=True, help='CUDA device number to set')
    #
    # args = parser.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    # print(f"CUDA_VISIBLE_DEVICES set to {args.cuda_device}")
    # COT = args.func
    pass

if __name__ == '__main__':
    # TODO
    model_name = "Llama3.1"
    inferllm = InferLLM(api_key="",
                        model_name=model_name)

    start_time = time.time()
    base_folder_path = 'filter_text/graphs_n5_t10/ER'
    base_output_path = "output/" + model_name + '/graphs_n5_t10/ER'
    # TODO
    COT = "EX_NOCODE"
    print(COT)
    if COT:
        base_output_path = base_output_path + "/" + COT
    if not os.path.exists(base_output_path):
        os.makedirs(base_output_path)

    inferllm.process_all_folders(base_folder_path, base_output_path, COT)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total processing time: {elapsed_time:.2f} seconds")
