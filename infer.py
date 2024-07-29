import os
import json
from LLM import LLaMA3_LLM
from tqdm import tqdm
import time
import re
from concurrent.futures import ThreadPoolExecutor

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

llm = LLaMA3_LLM(mode_name_or_path="/home/work/LargeModels/LLaMA3/LLM-Research/Meta-Llama-3-8B-Instruct")

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

uv_prompt = "(u,v,t) are equal to (v,u,t)."
cot_examples = {
    "When link": "For example, if you want to answer when are node 0 and node 2 first linked, you need to find the smallest time t such that the edge (0,2,t) or (2,0,t) exists in the graph.",
    "When connect": "For example, if you want to answer when are node 0 and node 2 first linked, you need to find the smallest time t such that there exists a path node 0 to node 2 or node 2 to node 0. A path from node 0 to node 2 is defined as a sequence of edges (0,a,t_0),(a,b,t_1),...,(d,2,t) and t_i<= t.",
    "When triadic closure": "For example, if you want to answer when do node 1,node 1 and node 2 first close the triad, you need to find the smallest time t such that the edges (0,1,t_1),(1,2,t_2) and (0,2,t_3) all exist with max(t_1,t_2,t_3) = t.",
    "What neighbors at time": "For example, if you want to answer what nodes are linked with node2 at time 3, you need to find the set of nodes N such the (2,N,3) or (N,2,3) exists in the graph.",
    "What neighbors in periods": "For example, if you want to answer what neighbors are linked with node 2 at or after time 1, you need to find the set of nodes N such that (2,N,t) or (N,2,t) exists in the graph for any t >= 1.",
    "Check triadic closure": "For example, if you want to answer whether nodes0,1 and 2 form a closed triad, you need to check if the edges (0,1,t_1),(1,2,t_2),(0,2,t_3) exist.",
    "Find temporal path": "For example, if you want to answer whether nodes 0,1 and 2 form a chronological path, you need to find a sequence of nodes {1,v_1,v_2,v_3,...,v_n} and corresponding times {t_0,t_1,...,t_n} such that t_0<= t_1 <=...<=t_n and edges(v_i,v_(i+1),t_i) exist in the graph.",
    "Sort edge by time": "For example, if you want to answer sort the edges by time from earliest to latest, you need to sort the edges in ascending order by the time t.",
    "Check temporal path": "For example, if you want to answer whether nodes 0,1,2 form a chronological path, you need to check if there exist the edges(0,1,t_0) and (1,2,_t0)."
}

text_examples = {
    "When link": "For example, if you want to answer when are node 0 and node 2 first linked, you need to find the smallest time t such that the edge (0,2,t) or (2,0,t) exists in the graph.",
    "When connect": "For example, if you want to answer when are node 0 and node 2 first linked, you need to find the smallest time t such that there exists a path node 0 to node 2 or node 2 to node 0. A path from node 0 to node 2 is defined as a sequence of edges (0,a,t_0),(a,b,t_1),...,(d,2,t) and t_i<= t.",
    "When triadic closure": "For example, if you want to answer when do node 1,node 1 and node 2 first close the triad, you need to find the smallest time t such that the edges (0,1,t_1),(1,2,t_2) and (0,2,t_3) all exist with max(t_1,t_2,t_3) = t.",
    "What neighbors at time": "For example, if you want to answer what nodes are linked with node2 at time 3, you need to find the set of nodes N such the (2,N,3) or (N,2,3) exists in the graph.",
    "What neighbors in periods": "For example, if you want to answer what neighbors are linked with node 2 at or after time 1, you need to find the set of nodes N such that (2,N,t) or (N,2,t) exists in the graph for any t >= 1.",
    "Check triadic closure": "For example, if you want to answer whether nodes0,1 and 2 form a closed triad, you need to check if the edges (0,1,t_1),(1,2,t_2),(0,2,t_3) exist.",
    "Find temporal path": "For example, if you want to answer whether nodes 0,1 and 2 form a chronological path, you need to find a sequence of nodes {1,v_1,v_2,v_3,...,v_n} and corresponding times {t_0,t_1,...,t_n} such that t_0<= t_1 <=...<=t_n and edges(v_i,v_(i+1),t_i) exist in the graph.",
    "Sort edge by time": "For example, if you want to answer sort the edges by time from earliest to latest, you need to sort the edges in ascending order by the time t.",
    "Check temporal path": "For example, if you want to answer whether nodes 0,1,2 form a chronological path, you need to check if there exist the edges(0,1,t_0) and (1,2,_t0)."
}

cyo_patch4link_forever = "As long as two nodes are linked,they will be linked forever."


def process_json_files_in_folder(input_folder_path, cot):
    results = []
    for filename in os.listdir(input_folder_path):
        if filename.endswith(".json"):
            input_file_path = os.path.join(input_folder_path, filename)
            with open(input_file_path, 'r', encoding='utf=8') as infile:
                data = json.load(infile)
            if "SIMPLE" in cot:
                answers = json2llama_cot(data)
            elif "COMPLEX" in cot:
                answers = json2llama_cot_special(data)
            elif "NO" in cot:
                answers = json2llama(data)
            elif "PATCH" in cot:
                answers = json2llama_patch(data)
            elif "EXPLAIN" in cot:
                answers = json2llama_explain(data)
            elif "INPUT" in cot:
                answers = json2llama_input(data)
            results.extend(answers)
    return results


def json2llama(data):
    results = []
    for item in tqdm(data, desc="Processing one json"):
        # zero_shot
        input_text = item["instruction"] + "\nNext, please answer:" + item[
            "input"] + " Explain the reason."
        reason = llm(input_text)
        llm.clear_history()
        input_text = item["instruction"] + "\nNext, please answer:" + item[
            "input"] + " Just output answer without any explanation!"
        response = llm(input_text)
        llm.clear_history()
        # one_shot
        oneshot_text = item["instruction"] + "\n" + oneshot_examples[item["task"]]
        llm(oneshot_text)
        oneshot_text = "\nNext, please answer:" + item[
            "input"] + " Just output answer without any explanation!"
        oneshot_response = llm(oneshot_text)
        llm.clear_history()
        # explain_one_shot
        oneshot_text = item["instruction"] + "\n" + oneshot_examples[
            item[
                "task"]] + "\nPlease Think step by step and translate the thought process for the above question, but only provide answers for the subsequent questions."
        llm(oneshot_text)
        oneshot_text = "\nNext, please answer:" + item[
            "input"] + " Just output answer without any explanation!"
        explain_response = llm(oneshot_text)
        llm.clear_history()
        results.append(
            {"task": item["task"], "input": input_text, "truth": item["output"], "zero_shot_output": response,
             "one_shot_output": oneshot_response, "explain_output": explain_response, "reason": reason})

    return results


def json2llama_cot(data):
    cot_edges = "Describe all the edges in the following problem, for example, (0,1,0) means nodes 0 and 1 are linked at time 0.\n"
    results = []
    for item in tqdm(data, desc="Processing one json"):
        # zero_shot+cot
        input_text = item["instruction"] + "\n" + cot_edges + item[
            "input"]
        describe = llm(input_text)
        reason_txt = item["instruction"] + "\nNext, please answer:" + item[
            "input"] + " Explain the reason."
        reason = llm(reason_txt)
        input_text = item["instruction"] + "\nNext, please answer:" + item[
            "input"] + " Just output answer without any explanation!"
        zero_cot_response = llm(input_text)
        llm.clear_history()
        # one_shot+cot
        input_text = item["instruction"] + "\n" + cot_edges + item[
            "input"]
        llm(input_text)
        oneshot_text = item["instruction"] + "\n" + oneshot_examples[item["task"]]
        llm(oneshot_text)
        oneshot_text = "\nNext, please answer:" + item[
            "input"] + " Just output answer without any explanation!"
        oneshot_response = llm(oneshot_text)
        llm.clear_history()
        results.append(
            {"task": item["task"], "input": input_text, "truth": item["output"], "zero_shot_output": zero_cot_response,
             "one_shot_output": oneshot_response, "describe": describe, "reason": reason})

    return results


def json2llama_cot_special(data):
    results = []
    for item in tqdm(data, desc="Processing one json"):
        # zero_shot+cot
        input_text = item["instruction"] + "\n" + cot_examples[item["task"]]
        llm(input_text)
        input_text = item["instruction"] + "\nNext, please answer:" + item[
            "input"] + " Just output answer without any explanation!"
        zero_cot_response = llm(input_text)
        llm.clear_history()
        # one_shot+cot
        input_text = item["instruction"] + "\n" + cot_examples[item["task"]]
        llm(input_text)
        oneshot_text = item["instruction"] + "\n" + oneshot_examples[item["task"]]
        llm(oneshot_text)
        oneshot_text = "\nNext, please answer:" + item[
            "input"] + " Just output answer without any explanation!"
        oneshot_response = llm(oneshot_text)
        llm.clear_history()
        results.append(
            {"task": item["task"], "input": input_text, "truth": item["output"], "zero_shot_output": zero_cot_response,
             "one_shot_output": oneshot_response})

    return results


def json2llama_patch(data):
    cot_edges = "Describe all the edges in the following problem, for example, (0,1,0) means nodes 0 and 1 are linked at time 0.\n"
    results = []
    for item in tqdm(data, desc="Processing one json"):
        # zero_shot+cot
        input_text = item["instruction"] + "\n" + cot_edges + cyo_patch4link_forever + item[
            "input"]
        llm(input_text)
        input_text = item["instruction"] + "\nNext, please answer:" + item[
            "input"] + " Just output answer without any explanation!"
        zero_cot_response = llm(input_text)
        llm.clear_history()
        # one_shot+cot
        input_text = item["instruction"] + "\n" + cot_edges + cyo_patch4link_forever + item[
            "input"]
        llm(input_text)
        oneshot_text = item["instruction"] + "\n" + oneshot_examples[item["task"]]
        llm(oneshot_text)
        oneshot_text = "\nNext, please answer:" + item[
            "input"] + " Just output answer without any explanation!"
        oneshot_response = llm(oneshot_text)
        llm.clear_history()
        results.append(
            {"task": item["task"], "input": input_text, "truth": item["output"], "zero_shot_output": zero_cot_response,
             "one_shot_output": oneshot_response})

    return results


def json2llama_explain(data):
    results = []
    for item in tqdm(data, desc="Processing one json"):
        # zero_shot+explain
        input_text = item["instruction"] + "\n" + item[
            "input"] + "\nPlease Think step by step and translate the thought process for the above question, but only provide answers for the subsequent questions."
        llm(input_text)
        input_text = item["instruction"] + "\nNext, please answer:" + item[
            "input"] + " Just output answer without any explanation!"
        response = llm(input_text)
        llm.clear_history()
        # one_shot+explain
        oneshot_text = item["instruction"] + "\n" + oneshot_examples[item["task"]]
        llm(oneshot_text)
        input_text = item["instruction"] + "\n" + item[
            "input"] + "\nPlease Think step by step and translate the thought process for the above question, but only provide answers for the subsequent questions."
        llm(input_text)
        oneshot_text = "\nNext, please answer:" + item[
            "input"] + " Just output answer without any explanation!"
        oneshot_response = llm(oneshot_text)
        llm.clear_history()
        results.append(
            {"task": item["task"], "input": input_text, "truth": item["output"], "zero_shot_output": response,
             "one_shot_output": oneshot_response})

    return results


def replace_triplet_and_edges(text):
    # 替换"Given an undirected dynamic graph with the edges [(...)]"
    text = re.sub(r"Given an undirected dynamic graph with the edges \[(.*?)\]",
                  lambda match: "Given an undirected dynamic graph with the edges : " + re.sub(
                      r"\((\d+), (\d+), (\d+)\)",
                      lambda m: f"node {m.group(1)} and node {m.group(2)} are linked at time {m.group(3)}",
                      match.group(1)),
                  text)
    return text


def json2llama_input(data):
    results = []
    for item in tqdm(data, desc="Processing one json"):
        new_input = replace_triplet_and_edges(item["input"])

        # zero_shot
        input_text = item[
                         "instruction"] + "\nNext, please answer:" + new_input + " Just output answer without any explanation!"
        response = llm(input_text)
        llm.clear_history()
        # one_shot
        oneshot_text = item["instruction"] + "\n" + oneshot_examples[item["task"]]
        llm(oneshot_text)
        oneshot_text = "\nNext, please answer:" + new_input + " Just output answer without any explanation!"
        oneshot_response = llm(oneshot_text)
        llm.clear_history()

        # zero_shot+patch
        input_text = item[
                         "instruction"] + cyo_patch4link_forever + "\nNext, please answer:" + new_input + " Just output answer without any explanation!"
        link_0 = llm(input_text)
        llm.clear_history()
        # one_shot+patch
        oneshot_text = item["instruction"] + cyo_patch4link_forever + "\n" + oneshot_examples[item["task"]]
        llm(oneshot_text)
        oneshot_text = "\nNext, please answer:" + new_input + " Just output answer without any explanation!"
        link_1 = llm(oneshot_text)
        llm.clear_history()
        results.append(
            {"task": item["task"], "input": input_text, "truth": item["output"], "zero_shot_output": response,
             "one_shot_output": oneshot_response, "link_0": link_0, "link_1": link_1})

    return results


def process_all_folders(base_folder_path, base_output_path, cot):
    for folder_name in tqdm(os.listdir(base_folder_path), desc=f"Processing folders"):
        print(folder_name)
        if "When link" not in folder_name:
            continue
        folder_path = os.path.join(base_folder_path, folder_name)
        if os.path.isdir(folder_path):
            output_file_path = os.path.join(base_output_path, folder_name + "_output.json")
            folder_result = process_json_files_in_folder(folder_path, cot)
            with open(output_file_path, 'w', encoding='utf-8') as outfile:
                json.dump(folder_result, outfile, ensure_ascii=False, indent=4)


start_time = time.time()
base_folder_path = '/home/work/chenyo/myPaper/text_data/graphs_n5_t5/ER'
base_output_path = '/home/work/chenyo/myPaper/onlyllama3/graphs_n5_t5/ER'
COT = "SIMPLE_when_link"
print(COT)
if COT:
    base_output_path = base_output_path + "/" + COT
if not os.path.exists(base_output_path):
    os.makedirs(base_output_path)

process_all_folders(base_folder_path, base_output_path, COT)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total processing time: {elapsed_time:.2f} seconds")
