import os
import pandas as pd
import random
import networkx as nx
import json


def generate_QA_When_link(edges, N):
    nodes = list(range(N))
    node1, node2 = random.sample(nodes, 2)

    edges_str = ", ".join([f"({src}, {tgt}, {time})" for src, tgt, time in edges])
    question = f"Given an undirected dynamic graph with the edges [{edges_str}]. When are node {node1} and node {node2} linked?"

    linked_time = None
    for src, tgt, time in edges:
        if (src == node1 and tgt == node2) or (src == node2 and tgt == node1):
            linked_time = time
            break

    if linked_time is not None:
        answer = f"{linked_time}"
    else:
        answer = f"-1"

    return question, answer


def generate_QA_When_connect(edges, N):
    node_range = list(range(N))
    node1, node2 = random.sample(node_range, 2)
    edges_str = ", ".join([f"({src}, {tgt}, {time})" for src, tgt, time in edges])
    question = f"Given an undirected dynamic graph with the edges [{edges_str}]. When are node {node1} and node {node2} first connected?"

    graph = nx.Graph()
    connected_time = None

    graph.add_nodes_from(list(range(N)))
    for time in sorted(set(time for _, _, time in edges)):
        for src, tgt, edge_time in edges:
            if edge_time == time:
                graph.add_edge(src, tgt)
                if nx.has_path(graph, node1, node2):
                    connected_time = time
                    break
        if connected_time is not None:
            break
    if connected_time is not None:
        answer = f"{connected_time}"
    else:
        answer = f"-1"

    return question, answer


def generate_QA_When_triadic_closure(edges, N):
    node_range = list(range(N))
    node1, node2, node3 = random.sample(node_range, 3)
    edges_str = ", ".join([f"({src}, {tgt}, {time})" for src, tgt, time in edges])
    question = f"Given an undirected dynamic graph with the edges [{edges_str}]. When are node {node1},{node2} and {node3} first close the triad?"

    graph = nx.Graph()
    triad_closed_time = None

    graph.add_nodes_from(list(range(N)))

    for time in sorted(set(time for _, _, time in edges)):
        for src, tgt, edge_time in edges:
            if edge_time == time:
                graph.add_edge(src, tgt)
                if node1 in graph.neighbors(node2) and node2 in graph.neighbors(node3) and node3 in graph.neighbors(
                        node1):
                    triad_closed_time = time
                    break
        if triad_closed_time is not None:
            break

    if triad_closed_time is not None:
        answer = f'{triad_closed_time}'
    else:
        answer = '-1'

    return question, answer


def generate_QA_What_neighbors_at_time(edges, N, T):
    node_range = list(range(N))
    node1 = random.choice(node_range)
    time1 = random.choice(list(range(T)))
    edges_str = ", ".join([f"({src}, {tgt}, {time})" for src, tgt, time in edges])
    question = f"Given an undirected dynamic graph with the edges [{edges_str}]. What nodes are linked with node {node1} at time {time1} ?"
    graph = nx.Graph()
    graph.add_nodes_from(list(range(N)))
    for src, tgt, edge_time in edges:
        if edge_time == time1:
            graph.add_edge(src, tgt)
    linked_nodes = list(graph.neighbors(node1))

    if linked_nodes is not None:
        answer = f'{linked_nodes}'
    else:
        answer = f'{[]}'
    return question, answer


def generate_QA_What_neighbors_in_periods(edges, N, T):
    node_range = list(range(N))
    node1 = random.choice(node_range)
    time1 = random.choice(list(range(T)))
    edges_str = ", ".join([f"({src}, {tgt}, {time})" for src, tgt, time in edges])
    question = f"Given an undirected dynamic graph with the edges [{edges_str}]. What nodes are linked with node {node1} at or after time {time1} ?"
    graph = nx.Graph()
    graph_before = nx.Graph()
    graph.add_nodes_from(list(range(N)))
    graph_before.add_nodes_from(list(range(N)))
    for src, tgt, edge_time in edges:
        if edge_time >= time1:
            graph.add_edge(src, tgt)
        else:
            graph_before.add_edge(src, tgt)
    linked_nodes = set(graph.neighbors(node1)) - set(graph_before.neighbors(node1))

    if linked_nodes is not None:
        answer = f'{list(linked_nodes)}'
    else:
        answer = f'{[]}'
    return question, answer


def get_last_two_folders(folder_path):
    normalized_path = os.path.normpath(folder_path)
    last_folder_name = os.path.basename(normalized_path)
    parent_folder_name = os.path.basename(os.path.dirname(normalized_path))
    return os.path.join(parent_folder_name, last_folder_name)


def process_files_in_folder(folder_path, N, T, question_list, instruction):
    current_dir = os.getcwd()
    last_folder_name = get_last_two_folders(folder_path)

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)

            edges = list(df.itertuples(index=False, name=None))
            nodes = set(df['Source']).union(set(df['Target']))

            for question_type in question_list:
                output_folder_path = os.path.join(current_dir, last_folder_name,
                                                  question_type)
                os.makedirs(output_folder_path, exist_ok=True)

                questions_answers = []
                for _ in range(10):
                    if question_type == "When link":
                        task_instruction = "Your task is to answer when two nodes are first linked in the dynamic graph.  Two nodes are linked if there exists a temporal edge between them."
                        answer_instruction = """ Give the answer as an integer number at the last of your response after 'Answer:' . If the answer does not exist, please respond with 'Answer: -1'."""
                        question, answer = generate_QA_When_link(edges, N)

                    elif question_type == "When connect":
                        task_instruction = """Your task is to answer when two nodes are first connected in the dynamic graph. Two nodes are connected if there exists a path between them."""
                        answer_instruction = """ Give the answer as an integer number at the last of your response after 'Answer:'. If the answer does not exist, please respond with 'Answer: -1'."""
                        question, answer = generate_QA_When_connect(edges, N)

                    elif question_type == "When triadic closure":
                        task_instruction = """Your task is to answer when when the three given nodes first form a closed triad. Two nodes with a common neighbor are said to have a triadic closure, if they are linked since some time so that the three nodes have linked with each other to form a triad."""
                        answer_instruction = """ Give the answer as an integer number at the last of your response after 'Answer:'. If the answer does not exist, please respond with 'Answer: -1'."""
                        question, answer = generate_QA_When_triadic_closure(edges, N)

                    elif question_type == "What neighbors at time":
                        task_instruction = """Your task is to answer what nodes are linked with a given node at a given time.  Two nodes are linked at a given time if there exists a temporal edge between them at the given time."""
                        answer_instruction = """ Give the answer as an array of integer number at the last of your response after 'Answer:'. If the answer does not exist, please respond with 'Answer: []'."""
                        question, answer = generate_QA_What_neighbors_at_time(edges, N, T)

                    elif question_type == "What neighbors in periods":
                        task_instruction = """Your task is to answer what nodes are linked with a given node after or at a given time,but not linked before the given time.Two nodes are linked after or at a given time if there exists a temporal edge between them after or at the given time."""
                        answer_instruction = """ Give the answer as an array of integer number at the last of your response after 'Answer:'. If the answer does not exist, please respond with 'Answer: []'."""
                        question, answer = generate_QA_What_neighbors_in_periods(edges, N, T)
                    else:
                        raise ValueError(f"Unknown question type: {question_type}")
                    questions_answers.append(
                        {"task": question_type,
                         "instruction": instruction + task_instruction,
                         "input": question + answer_instruction,
                         "output": "Answer: " + answer})

                output_file_path = os.path.join(output_folder_path, filename.replace(".csv", ".json"))
                with open(output_file_path, 'w') as f:
                    json.dump(questions_answers, f, indent=4)


myN = 10
myT = 20
gtype = "ER"
p = 0.5

name = f"n{myN}_t{myT}_p{p}"

my_folder_path = f'../graph_data/graphs_{name}/{gtype}'

all_question_types = [
    "When link",
    "When connect",
    "When triadic closure",
    "What neighbors at time",
    "What neighbors in periods",
]

myInstruction = "In an undirected dynamic graph, (u, v, t) means that node u and node v are linked with an undirected edge at time t."

process_files_in_folder(my_folder_path, myN, myT, all_question_types, myInstruction)
