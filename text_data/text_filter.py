import os
import pandas as pd
import random
import networkx as nx
import json

myN = 10
myT = 20
p = 0.5

gtype = "ER"
name = f"n{myN}_t{myT}_p{p}"
my_folder_path = f'../text_data/graphs_{name}/{gtype}'
output_path = f'../filter_text/graphs_{name}/{gtype}'


def save_to_json(data, filename):
    try:
        filename = filename + '.json'
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error")


answer_type = {
    "When link": "Answer: -1",
    "When connect": "Answer: -1",
    "When triadic closure": "Answer: -1",
    "What neighbors at time": "Answer: []",
    "What neighbors in periods": "Answer: []",
}

for filename in os.listdir(my_folder_path):
    merged_data = []
    task_path = os.path.join(my_folder_path, filename)
    for graph in os.listdir(task_path):
        graph_path = os.path.join(task_path, graph)
        with open(graph_path, 'r') as file:
            data = json.load(file)
            merged_data.extend(data)
    print(filename)

    unique_data = {}
    for item in merged_data:
        unique_data[item['input']] = item
    merged_data = list(unique_data.values())
    random.shuffle(merged_data)
    no_answer_data = [item for item in merged_data if item.get('output') == answer_type[filename]]
    answer_data = [item for item in merged_data if item.get('output') != answer_type[filename]]
    print("no answer:" + str(len(no_answer_data)))
    print("have answer:" + str(len(answer_data)))

    no_answer_data = no_answer_data[:100]
    answer_data = answer_data[:100]
    no_answer_path = os.path.join(output_path, 'no_answer', filename)
    answer_path = os.path.join(output_path, 'have_answer', filename)
    save_to_json(no_answer_data, no_answer_path)
    save_to_json(answer_data, answer_path)
