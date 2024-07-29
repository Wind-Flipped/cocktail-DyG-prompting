import os
import json
# from LLM import LLaMA3_LLM, Qwen2_LLM, Phi3Small_LLM, Gemma2_LLM, ChatGLM4_LLM, Yi_LLM
from apiLLM import Phi3Small_LLM, LLM
from tqdm import tqdm
import time
import re
from concurrent.futures import ThreadPoolExecutor
import argparse

# TODO
model = "Llama3.1"
print(model)

if model == "Phi3":
    llm = Phi3Small_LLM()
elif model == "Llama3.1":
    llm = LLM(model_name="meta-llama/llama-3.1-8b-instruct")
elif model == "Mistral":
    llm = LLM(model_name="mistralai/mistral-7b-instruct")
elif model == "Gpt-3.5":
    llm = LLM(model_name="openai/gpt-3.5-turbo-0125")
elif model == "Claude-3":
    llm = LLM(model_name="anthropic/claude-3-haiku")
else:
    llm = LLM()

# if "LLAMA3" == model:
#     llm = LLaMA3_LLM(mode_name_or_path="/home/work/LargeModels/LLaMA3/LLM-Research/Meta-Llama-3-8B-Instruct")
# elif model == "Qwen2":
#     llm = Qwen2_LLM(mode_name_or_path="/home/work/chenyo/myPaper/model/qwen/Qwen2-7B-Instruct")
# elif model == "Phi3":
#     llm = Phi3Small_LLM(mode_name_or_path="/home/work/chenyo/myPaper/model/LLM-Research/Phi-3-small-8k-instruct")
# elif model == "GLM4":
#     llm = ChatGLM4_LLM(mode_name_or_path="/home/work/chenyo/myPaper/model/LLM-Research/ZhipuAI/glm-4-9b-chat")
# elif model == "Yi":
#     llm = Yi_LLM(mode_name_or_path="/home/work/chenyo/myPaper/model/01ai/Yi-6B-Chat")
# else:
#     llm = Gemma2_LLM(mode_name_or_path="/home/work/chenyo/myPaper/model/LLM-Research/gemma-2-9b-it")

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
# cot4no_answer
check = ""
afraid = ""
encouragement = ""
explain_format = "Just output in JSON format, containing two fields: Answer and Reason.The Answer field contains only the answer, and the Reason field contains your explanation."
no_explain_format = " Just output answer without any explanation!"
output_format = no_explain_format


def process_json_files(file_path, cot):
    results = []
    with open(file_path, 'r', encoding='utf=8') as infile:
        data = json.load(infile)
    if "EDGE" == cot:
        answers = json2llama_edge(data)
        results.extend(answers)
    elif "NO" == cot:
        answers = json2llama(data)
        results.extend(answers)
    elif "PATCH" == cot:
        answers = json2llama_patch(data)
        results.extend(answers)
    elif "CHECK" == cot:
        answers = json2llama_check(data)
        results.extend(answers)
    elif "EXPLAIN" == cot:
        answers = json2llama_explain(data)
        results.extend(answers)
    elif "RECONFIRM" == cot:
        answers = json2llama_reconfirm(data)
        results.extend(answers)
    elif "ADD" == cot:
        answers = json2llama_add(data)
        results.extend(answers)
    elif "ASSUMPTION" == cot:
        answers = json2llama_cur(data)
        results.extend(answers)
    elif "EX_V_CUR" == cot:
        answers = json2llama_ex_v_cur(data)
        results.extend(answers)
    elif "EX_CUR" == cot:
        answers = json2llama_ex_cur(data)
        results.extend(answers)
    elif "EX_CHECK" == cot:
        answers = json2llama_ex_check(data)
        results.extend(answers)
    elif "EX_ALL" == cot:
        answers = json2llama_ex_all(data)
        results.extend(answers)
    elif "ALL" == cot:
        answers = json2llama_all(data)
        results.extend(answers)
    elif "CONNECT_LINK" == cot:
        answers = json2llama_ex_link_connect(data)
        results.extend(answers)
    elif "EX_EDGE" == cot:
        answers = json2llama_ex_edge(data)
        results.extend(answers)
    elif "EXPLAIN1" == cot:
        answers = json2llama_explain1(data)
        results.extend(answers)
    elif "EXPLAIN2" == cot:
        answers = json2llama_explain2(data)
        results.extend(answers)
    elif "EX_ALL1" == cot:
        answers = json2llama_ex_all1(data)
        results.extend(answers)
    return results


# no
def json2llama(data):
    results = []
    for item in tqdm(data, desc="Processing one json"):
        input_text = item["instruction"] + "\nNext, please answer: " + item[
            "input"] + output_format
        zero_response = llm(input_text)
        llm.clear_history()
        results.append(
            {"task": item["task"], "input": input_text, "truth": item["output"], "zero_shot_output": zero_response})

    return results


def json2llama_ex_edge(data):
    cot_edges = "Describe all the edges in the following problem, for example, (0,1,0) means nodes 0 and 1 are linked at time 0.\n"
    results = []
    for item in tqdm(data, desc="Processing one json"):
        # zero_shot+explain
        input_text = item["instruction"] + cot_edges + item[
            "input"]
        describe0 = llm(input_text)
        input_text = item["instruction"] + "\n" + item[
            "input"] + "\nPlease explain the reason in detail."
        describe = llm(input_text)
        input_text = "Give the answer to the next question without explanation." + item[
            "instruction"] + "\nNext, please answer:" + item[
                         "input"] + " Just output answer without any explanation!" + "Ensure that your answer matches your explanation."
        zero_response = llm(input_text)
        llm.clear_history()
        results.append(
            {"task": item["task"], "input": input_text, "truth": item["output"], "zero_shot_output": zero_response,
             "describe": describe, "describe0": describe0})

    return results


# explain
def json2llama_explain(data):
    results = []
    for item in tqdm(data, desc="Processing one json"):
        # zero_shot+explain
        input_text = item["instruction"] + "\n" + item[
            "input"] + "\nPlease explain the reason in detail."
        describe = llm(input_text)
        input_text = "Give the answer to the next question without explanation." + item[
            "instruction"] + "\nNext, please answer: " + item[
                         "input"] + " Just output answer without any explanation!"
        zero_response = llm(input_text)
        llm.clear_history()
        results.append(
            {"task": item["task"], "input": input_text, "truth": item["output"], "zero_shot_output": zero_response,
             "describe": describe})

    return results


def json2llama_explain1(data):
    results = []
    for item in tqdm(data, desc="Processing one json"):
        # zero_shot+explain
        input_text = item["instruction"] + "\n" + item[
            "input"] + "\nPlease  explain the reasons in detail."
        describe = llm(input_text)
        input_text = "Give the answer to the next question without explanation." + item[
            "instruction"] + "\nNext, please answer: " + item[
                         "input"] + " Just output answer without any explanation!"
        zero_response = llm(input_text)
        llm.clear_history()
        results.append(
            {"task": item["task"], "input": input_text, "truth": item["output"], "zero_shot_output": zero_response,
             "describe": describe})

    return results


def json2llama_explain2(data):
    results = []
    for item in tqdm(data, desc="Processing one json"):
        # zero_shot+explain
        input_text = item["instruction"] + "\n" + item[
            "input"] + "\nPlease think step by step."
        describe = llm(input_text)
        input_text = "Give the answer to the next question without explanation." + item[
            "instruction"] + "\nNext, please answer: " + item[
                         "input"] + " Just output answer without any explanation!"
        zero_response = llm(input_text)
        llm.clear_history()
        results.append(
            {"task": item["task"], "input": input_text, "truth": item["output"], "zero_shot_output": zero_response,
             "describe": describe})

    return results


# expalin+reconfirm
def json2llama_reconfirm(data):
    results = []
    for item in tqdm(data, desc="Processing one json"):
        # zero_shot+explain
        input_text = item["instruction"] + "\n" + item[
            "input"] + "\nPlease explain the reason in detail."
        describe = llm(input_text)
        input_text = "Give the answer to the next question without explanation." + item[
            "instruction"] + "\nNext, please answer: " + item[
                         "input"] + " Just output answer without any explanation!" + "Ensure that your answer matches your explanation."
        zero_response = llm(input_text)
        llm.clear_history()
        results.append(
            {"task": item["task"], "input": input_text, "truth": item["output"], "zero_shot_output": zero_response,
             "describe": describe})

    return results


def json2llama_ex_link_connect(data):
    results = []
    link_connect = "Two nodes are linked if and only if there is a direct edge between them. Two nodes are connected if and only if they can be indirectly connected through other nodes, meaning there exists a path between them."
    for item in tqdm(data, desc="Processing one json"):
        # zero_shot+explain
        input_text = item["instruction"] + "\n" + link_connect + item[
            "input"] + "\nPlease explain the reason in detail."
        describe = llm(input_text)
        input_text = "Give the answer to the next question without explanation." + item[
            "instruction"] + "\nNext, please answer: " + item[
                         "input"] + " Just output answer without any explanation!" + "Ensure that your answer matches your explanation."
        zero_response = llm(input_text)
        llm.clear_history()
        results.append(
            {"task": item["task"], "input": input_text, "truth": item["output"], "zero_shot_output": zero_response,
             "describe": describe})

    return results


# explain+no_assumption
def json2llama_ex_cur(data):
    no_assumption = "Do not make any assumptions, just focus on the current conditions."
    results = []
    for item in tqdm(data, desc="Processing one json"):
        input_text = item["instruction"] + "\n" + item[
            "input"] + "\nPlease explain the reason in detail." + no_assumption
        describe = llm(input_text)
        input_text = "Give the answer to the next question without explanation." + item[
            "instruction"] + "\nNext, please answer: " + item[
                         "input"] + " Just output answer without any explanation!"
        zero_response = llm(input_text)
        llm.clear_history()
        results.append(
            {"task": item["task"], "input": input_text, "truth": item["output"], "zero_shot_output": zero_response,
             "describe": describe})
    return results


# explain+check
def json2llama_ex_check(data):
    check_nt = "Verify that the nodes,time and edges in the problem do indeed exist."
    results = []
    for item in tqdm(data, desc="Processing one json"):
        input_text = item["instruction"] + "\n" + item[
            "input"] + "\nPlease explain the reason in detail." + check_nt
        describe = llm(input_text)
        input_text = "Give the answer to the next question without explanation." + item[
            "instruction"] + "\nNext, please answer:" + item[
                         "input"] + " Just output answer without any explanation!"
        zero_response = llm(input_text)
        llm.clear_history()
        results.append(
            {"task": item["task"], "input": input_text, "truth": item["output"], "zero_shot_output": zero_response,
             "describe": describe})

    return results


# explain+all
def json2llama_ex_all(data):
    check_nt = "Verify that the nodes,time and edges in the problem do indeed exist."
    results = []
    no_assumption = "Do not make any assumptions, just focus on the current conditions."
    for item in tqdm(data, desc="Processing one json"):
        # zero_shot+explain
        input_text = item["instruction"] + "\n" + item[
            "input"] + "\nPlease explain the reason in detail." + check_nt + no_assumption
        describe = llm(input_text)
        input_text = "Give the answer to the next question without explanation." + item[
            "instruction"] + "\nNext, please answer: " + item[
                         "input"] + " Just output answer without any explanation!" + "Ensure that your answer matches your explanation."
        zero_response = llm(input_text)
        llm.clear_history()
        results.append(
            {"task": item["task"], "input": input_text, "truth": item["output"], "zero_shot_output": zero_response,
             "describe": describe})

    return results


def json2llama_ex_all1(data):
    check_nt = "Verify that the nodes,time and edges in the problem do indeed exist."
    results = []
    no_assumption = "Do not make any assumptions, just focus on the current conditions."
    for item in tqdm(data, desc="Processing one json"):
        # zero_shot+explain
        input_text = item["instruction"] + "\n" + item[
            "input"] + "\nPlease explain the reason."
        describe = llm(input_text)
        llm(check_nt)
        llm(no_assumption)
        input_text = "Give the answer to the next question without explanation." + item[
            "instruction"] + "\nNext, please answer: " + item[
                         "input"] + " Just output answer without any explanation!" + "Ensure that your answer matches your explanation."
        zero_response = llm(input_text)
        llm.clear_history()
        results.append(
            {"task": item["task"], "input": input_text, "truth": item["output"], "zero_shot_output": zero_response,
             "describe": describe})

    return results


# check
def json2llama_check(data):
    check_nt = "Verify that the nodes,time and edges in the problem do indeed exist."
    results = []
    for item in tqdm(data, desc="Processing one json"):
        input_text = item["instruction"] + check_nt + item[
            "input"]
        describe = llm(input_text)
        input_text = item["instruction"] + "\nNext, please answer:" + item[
            "input"] + output_format
        zero_response = llm(input_text)
        llm.clear_history()
        results.append(
            {"task": item["task"], "input": input_text, "truth": item["output"], "zero_shot_output": zero_response,
             "describe": describe})

    return results


def json2llama_ex_v_cur(data):
    results = []
    no_assumption = "Do not make any assumptions, just focus on the current conditions."
    for item in tqdm(data, desc="Processing one json"):
        # zero_shot+explain

        input_text = item["instruction"] + "\n" + item[
            "input"] + "\nPlease explain the reason in detail." + no_assumption
        describe = llm(input_text)
        input_text = "Give the answer to the next question without explanation." + item[
            "instruction"] + "\nNext, please answer:" + item[
                         "input"] + " Just output answer without any explanation!" + "Ensure that your answer matches your explanation."
        zero_response = llm(input_text)
        llm.clear_history()
        results.append(
            {"task": item["task"], "input": input_text, "truth": item["output"], "zero_shot_output": zero_response,
             "describe": describe})

    return results


def json2llama_cur(data):
    no_assumption = "Do not make any assumptions, just focus on the current conditions."
    results = []
    for item in tqdm(data, desc="Processing one json"):
        input_text = item["instruction"] + no_assumption + item[
            "input"]
        describe = llm(input_text)
        input_text = item["instruction"] + "\nNext, please answer:" + item[
            "input"] + output_format
        zero_response = llm(input_text)
        llm.clear_history()
        results.append(
            {"task": item["task"], "input": input_text, "truth": item["output"], "zero_shot_output": zero_response,
             "describe": describe})
    return results


def json2llama_edge(data):
    cot_edges = "Describe all the edges in the following problem, for example, (0,1,0) means nodes 0 and 1 are linked at time 0.\n"
    results = []
    for item in tqdm(data, desc="Processing one json"):
        input_text = item["instruction"] + cot_edges + item[
            "input"]
        describe = llm(input_text)
        input_text = item["instruction"] + "\nNext, please answer:" + item[
            "input"] + output_format
        zero_response = llm(input_text)
        llm.clear_history()
        results.append(
            {"task": item["task"], "input": input_text, "truth": item["output"], "zero_shot_output": zero_response,
             "describe": describe})

    return results


def json2llama_patch(data):
    patch4link_forever = "As long as two nodes are linked,they will be linked forever."
    results = []
    for item in tqdm(data, desc="Processing one json"):
        input_text = item["instruction"] + patch4link_forever + item[
            "input"]
        describe = llm(input_text)
        input_text = item["instruction"] + "\nNext, please answer:" + item[
            "input"] + output_format
        zero_response = llm(input_text)
        llm.clear_history()
        results.append(
            {"task": item["task"], "input": input_text, "truth": item["output"], "zero_shot_output": zero_response,
             "describe": describe})

    return results


def json2llama_add(data):
    cot_edges = "Describe all the edges in the following problem, for example, (0,1,0) means nodes 0 and 1 are linked at time 0.\n"
    check_nt = "Verify that the nodes,time and edges in the problem do indeed exist."
    results = []
    for item in tqdm(data, desc="Processing one json"):
        input_text = item["instruction"] + check_nt + item[
            "input"]
        describe = llm(input_text)
        input_text = item["instruction"] + cot_edges + item[
            "input"]
        describe1 = llm(input_text)
        input_text = item["instruction"] + "\nNext, please answer:" + item[
            "input"] + output_format
        zero_response = llm(input_text)
        llm.clear_history()
        results.append(
            {"task": item["task"], "input": input_text, "truth": item["output"], "zero_shot_output": zero_response,
             "describe": describe, "describe1": describe1})
    return results


def json2llama_all(data):
    cot_edges = "Describe all the edges in the following problem, for example, (0,1,0) means nodes 0 and 1 are linked at time 0.\n"
    check_nt = "Verify that the nodes,time and edges in the problem do indeed exist."
    results = []
    no_assumption = "Do not make any assumptions, just focus on the current conditions."
    for item in tqdm(data, desc="Processing one json"):
        # zero_shot+explain
        input_text = item["instruction"] + cot_edges + item[
            "input"]
        describe0 = llm(input_text)
        input_text = item["instruction"] + "\n" + item[
            "input"] + "\nPlease explain the reason in detail." + check_nt + no_assumption
        describe = llm(input_text)
        input_text = "Give the answer to the next question without explanation." + item[
            "instruction"] + "\nNext, please answer:" + item[
                         "input"] + " Just output answer without any explanation!" + "Ensure that your answer matches your explanation."
        zero_response = llm(input_text)
        llm.clear_history()
        results.append(
            {"task": item["task"], "input": input_text, "truth": item["output"], "zero_shot_output": zero_response,
             "describe": describe, "describe0": describe0})

    return results


def process_all_folders(base_folder, base_output, cot):
    answer_types = ['have_answer', 'no_answer']
    for answer_type in answer_types:
        file_path = base_folder + '/' + answer_type
        for file_name in tqdm(os.listdir(file_path), desc=f"Processing folders"):

            print(answer_type + file_name)
            if os.path.isdir(file_path):
                output_file_path = os.path.join(base_output, answer_type)
                if not os.path.isdir(output_file_path):
                    os.mkdir(output_file_path)
                result = process_json_files(file_path + '/' + file_name, cot)
                # result = []
                with open(output_file_path + '/' + file_name, 'w', encoding='utf-8') as outfile:
                    json.dump(result, outfile, ensure_ascii=False, indent=4)


def main():
    # parser = argparse.ArgumentParser(description="myPaper")
    #
    # # 添加参数
    # parser.add_argument('--func', type=str, required=True)
    # # parser.add_argument('--cuda_device', type=str, required=True, help='CUDA device number to set')
    #
    # # 解析参数
    # args = parser.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    # print(f"CUDA_VISIBLE_DEVICES set to {args.cuda_device}")
    # COT = args.func

    start_time = time.time()
    base_folder_path = 'filter_text/graphs_n5_t10/ER'
    base_output_path = "output/" + model + '/graphs_n5_t10/ER'
    # TODO
    COT = "EDGE"
    print(COT)
    if COT:
        base_output_path = base_output_path + "/" + COT
    if not os.path.exists(base_output_path):
        os.makedirs(base_output_path)

    process_all_folders(base_folder_path, base_output_path, COT)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total processing time: {elapsed_time:.2f} seconds")


if __name__ == '__main__':
    main()
