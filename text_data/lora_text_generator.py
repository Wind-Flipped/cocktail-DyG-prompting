import os
import pandas as pd
import random
import networkx as nx
import json


def process_files_in_folder(folder_path, N, T, instruction):
    current_dir = os.getcwd()  # 获取当前工作目录
    edges_explanation = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            edges = list(df.itertuples(index=False, name=None))
            edges_str = ", ".join([f"({src}, {tgt}, {time})" for src, tgt, time in edges])
            node_str = ", ".join(f"{num}" for num in range(N))
            lora_input = instruction + f"Given an undirected dynamic graph with the edges [{edges_str}] and nodes {node_str}"
            lora_output = " ".join(
                [
                    f"({src}, {tgt}, {time}) and ({tgt}, {src}, {time}) are equivalent, meaning that node {src} and node {tgt} are linked at time {time}."
                    for src, tgt, time in
                    edges])
            edges_explanation.append({
                "instruction": "Describe all the edges.For example, \"(0, 1, 2) and (1, 0, 2) are equivalent, meaning that node 0 and node 1 are linked at time 2\".",
                "input": lora_input,
                "output": lora_output
            })
            output_folder_path = os.path.join(current_dir, "lora")  # 在当前目录下创建包含最后一段目录名和question_type的文件夹
            os.makedirs(output_folder_path, exist_ok=True)
            output_file_path = os.path.join(output_folder_path, f"edges_explanation4n{N}T{T}.json")
            with open(output_file_path, 'w') as f:
                json.dump(edges_explanation, f, indent=4)


myN = 5
myT = 5
gtype = "ER"
# 文件夹路径
my_folder_path = f'../graph_data/graphs_n{myN}_t{myT}/{gtype}'

myInstruction = "In an undirected dynamic graph, (u, v, t) means that node u and node v are linked with an undirected edge at time t."

process_files_in_folder(my_folder_path, myN, myT, myInstruction)
