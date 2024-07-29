import os
import pandas as pd
import random
import networkx as nx
import json


def generate_QA_When_link(edges, N):
    # 随机选择两个不同的节点
    nodes = list(range(N))
    node1, node2 = random.sample(nodes, 2)

    # 生成问题
    edges_str = ", ".join([f"({src}, {tgt}, {time})" for src, tgt, time in edges])
    question = f"Given an undirected dynamic graph with the edges [{edges_str}]. When are node {node1} and node {node2} linked?"

    # 找到两个节点第一次连接的时间
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
    node_range = list(range(N))  # 从0到4选择节点
    node1, node2 = random.sample(node_range, 2)
    edges_str = ", ".join([f"({src}, {tgt}, {time})" for src, tgt, time in edges])
    question = f"Given an undirected dynamic graph with the edges [{edges_str}]. When are node {node1} and node {node2} first connected?"

    # 创建一个动态图，并逐步添加边来检测节点连接时间
    graph = nx.Graph()
    connected_time = None
    # 先添加所有的节点到图中
    graph.add_nodes_from(list(range(5)))
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
    node_range = list(range(N))  # 从0到4选择节点
    node1, node2, node3 = random.sample(node_range, 3)
    edges_str = ", ".join([f"({src}, {tgt}, {time})" for src, tgt, time in edges])
    question = f"Given an undirected dynamic graph with the edges [{edges_str}]. When are node {node1},{node2} and {node3} first close the triad?"
    # 创建一个动态图，并逐步添加边来检测节点连接时间
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
    node_range = list(range(N))  # 从0到4选择节点
    node1 = random.choice(node_range)
    time1 = random.choice(list(range(T)))
    edges_str = ", ".join([f"({src}, {tgt}, {time})" for src, tgt, time in edges])
    question = f"Given an undirected dynamic graph with the edges [{edges_str}]. What nodes are linked with node {node1} at time {time1} ?"
    # 创建一个动态图，并逐步添加边来检测节点连接时间
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
    node_range = list(range(N))  # 从0到4选择节点
    node1 = random.choice(node_range)
    time1 = random.choice(list(range(T)))
    edges_str = ", ".join([f"({src}, {tgt}, {time})" for src, tgt, time in edges])
    question = f"Given an undirected dynamic graph with the edges [{edges_str}]. What nodes are linked with node {node1} at or after time {time1} ?"
    # 创建一个动态图，并逐步添加边来检测节点连接时间
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


def generate_Check_triadic_closure(edges, N):
    node_range = list(range(N))  # 从0到4选择节点
    node1, node2, node3 = random.sample(node_range, 3)
    edges_str = ", ".join([f"({src}, {tgt}, {time})" for src, tgt, time in edges])
    question = f"Given an undirected dynamic graph with the edges [{edges_str}].  Did node {node1},{node2} and {node3} from a closed triad?"
    # 创建一个动态图，并逐步添加边来检测节点连接时间
    graph = nx.Graph()
    triad_closed = 'No'

    graph.add_nodes_from(list(range(N)))

    for src, tgt, edge_time in edges:
        graph.add_edge(src, tgt)
        if node1 in graph.neighbors(node2) and node2 in graph.neighbors(node3) and node3 in graph.neighbors(
                node1):
            triad_closed = 'Yes'
            break

    return question, triad_closed


def generate_Check_temporal_path(edges, N):
    target_nodes = random.sample(range(5), 3)  # 从节点中随机采样目标节点
    target_nodes_str = ', '.join(str(node) for node in target_nodes)
    edges_str = ", ".join([f"({src}, {tgt}, {time})" for src, tgt, time in edges])
    question = f"Given an undirected dynamic graph with the edges [{edges_str}]. Did nodes {target_nodes_str} form a chronological path?"

    # Create a dynamic graph and add edges to check for a chronological path
    graph = nx.Graph()

    # Add source and target nodes to the graph
    graph.add_nodes_from(range(N))

    for src, tgt, edge_time in sorted(edges, key=lambda x: x[2]):
        graph.add_edge(src, tgt, time=edge_time)
        # If all target nodes are connected in order, it forms a chronological path
        paths = list(
            nx.algorithms.simple_paths.all_simple_paths(graph, source=target_nodes[0], target=target_nodes[-1]))
        if any(
                path == target_nodes and
                all(graph.edges[(edge, next_edge)]['time'] <= graph.edges[(next_edge, next_next_edge)]['time'] for
                    edge, next_edge, next_next_edge in zip(path[:-2], path[1:-1], path[2:]))
                for path in paths
        ):
            return question, f"Yes"

    return question, f"No"


def generate_Find_temporal_path(edges, N):
    node1 = random.choice(range(N))  # 从节点中随机采样目标节点
    edges_str = ", ".join([f"({src}, {tgt}, {time})" for src, tgt, time in edges])
    question = f"Given an undirected dynamic graph with the edges [{edges_str}]. Find a chronological path starting from node {node1}. "
    graph = nx.Graph()
    graph.add_nodes_from(range(N))
    for u, v, t in edges:
        graph.add_edge(u, v, time=t)
    sorted_edges = sorted(graph.edges(data=True), key=lambda x: x[2]['time'])

    def dfs(current_node, current_time, path):
        paths = [path]
        for u, v, data in sorted_edges:
            if data['time'] >= current_time:
                if u == current_node and v not in path:
                    new_paths = dfs(v, data['time'], path + [v])
                    paths.extend(new_paths)
                elif v == current_node and u not in path:
                    new_paths = dfs(u, data['time'], path + [u])
                    paths.extend(new_paths)
        return paths

    all_paths = dfs(node1, -1, [node1])

    if len(all_paths) == 1 and len(all_paths[0]) == 1:
        return question, "[]"
    else:
        return question, f"{all_paths}"


def generate_Sort_edge_by_time(edges):
    edges_str = ", ".join([f"({src}, {tgt}, {time})" for src, tgt, time in edges])
    question = f"Given an undirected dynamic graph with the edges [{edges_str}].  Sort the edges by time from earliest to latest. "
    sorted_edges = sorted(edges, key=lambda x: x[2])
    sorted_edges_str = ", ".join([f"({src}, {tgt}, {time})" for src, tgt, time in sorted_edges])
    return question, f"[{sorted_edges_str}]"


def get_last_two_folders(folder_path):
    # 使用os.path.normpath将路径标准化，os.path.split分割路径
    normalized_path = os.path.normpath(folder_path)
    last_folder_name = os.path.basename(normalized_path)
    parent_folder_name = os.path.basename(os.path.dirname(normalized_path))

    # 返回最后两段目录名
    return os.path.join(parent_folder_name, last_folder_name)


def process_files_in_folder(folder_path, N, T, question_list, instruction):
    current_dir = os.getcwd()  # 获取当前工作目录
    last_folder_name = get_last_two_folders(folder_path)  # 获取folder_path的最后一段目录名

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)

            edges = list(df.itertuples(index=False, name=None))
            nodes = set(df['Source']).union(set(df['Target']))

            for question_type in question_list:
                output_folder_path = os.path.join(current_dir, last_folder_name,
                                                  question_type)  # 在当前目录下创建包含最后一段目录名和question_type的文件夹
                os.makedirs(output_folder_path, exist_ok=True)

                questions_answers = []
                for _ in range(2):
                    if question_type == "When link":
                        task_instruction = "Your task is to answer when two nodes are first connected in the dynamic graph.  two nodes are linked if there exists a temporal edge between them."
                        answer_instruction = """ Give the answer as an integer number at the last of your response after 'Answer:' .If the answer does not exist, please respond with 'Answer: -1'."""
                        question, answer = generate_QA_When_link(edges, N)

                    elif question_type == "When connect":
                        task_instruction = """Your task is to answer when two nodes are first connected in the dynamic graph. Two nodes are connected if there exists a path between them."""
                        answer_instruction = """ Give the answer as an integer number at the last of your response after 'Answer:'.If the answer does not exist, please respond with 'Answer: -1'."""
                        question, answer = generate_QA_When_connect(edges, N)

                    elif question_type == "When triadic closure":
                        task_instruction = """Your task is to answer when when the three given nodes first form a closed triad. two nodes with a common neighbor are said to have a triadic closure, if they are linked since some time so that the three nodes have linked with each other to form a triad."""
                        answer_instruction = """ Give the answer as an integer number at the last of your response after 'Answer:'If the answer does not exist, please respond with 'Answer: -1'."""
                        question, answer = generate_QA_When_triadic_closure(edges, N)

                    elif question_type == "What neighbors at time":
                        task_instruction = """Your task is to answer what nodes are linked with a given node at a given time.  two nodes are linked at a given time if there exists a temporal edge between them at the given time."""
                        answer_instruction = """ Give the answer as an array of integer number at the last of your response after 'Answer:'If the answer does not exist, please respond with 'Answer: []'."""
                        question, answer = generate_QA_What_neighbors_at_time(edges, N, T)

                    elif question_type == "What neighbors in periods":
                        task_instruction = """Your task is to answer what nodes are linked with a given node after or at a given time,but not linked before the given time.two nodes are linked after or at a given time if there exists a temporal edge between them after or at the given time."""
                        answer_instruction = """ Give the answer as an array of integer number at the last of your response after 'Answer:'If the answer does not exist, please respond with 'Answer: []'."""
                        question, answer = generate_QA_What_neighbors_in_periods(edges, N, T)

                    elif question_type == "Check triadic closure":
                        task_instruction = """Your task is to answer whether the three given nodes form a closed triad. two nodes with a common neighbor are said to have a triadic closure, if they are linked since some time so that the three nodes have linked with each other to form a triad."""
                        answer_instruction = """ Give the answer as 'Yes' or 'No' after 'Answer:'."""
                        question, answer = generate_Check_triadic_closure(edges, N)

                    elif question_type == "Check temporal path":
                        task_instruction = """Your task is to answer whether the given three ordered nodes form a chronological path. A sequence of nodes construct a chronological path if the timestamps of the edges do not decrease from source node to target node in the path."""
                        answer_instruction = """ Give the answer as 'Yes' or 'No' after 'Answer:'."""
                        question, answer = generate_Check_temporal_path(edges, N)

                    elif question_type == "Find temporal path":
                        task_instruction = """Your task is to answer find a chronological path starting from a given node in the dynamic graph. A sequence of nodes construct a chronological path if the timestamps of the edges do not decrease from source node to target node in the path."""
                        answer_instruction = """Give the answer as an array of integer number at the last of your response after 'Answer:'If the answer does not exist, please respond with 'Answer: []'."""
                        question, answer = generate_Find_temporal_path(edges, N)

                    elif question_type == "Sort edge by time":
                        task_instruction = """Your task is to  sort the edges by time from earliest to latest."""
                        answer_instruction = """Give the answer as an array of (u, v, t) at the last of your response after 'Answer:'If the answer does not exist, please respond with 'Answer: []'."""
                        question, answer = generate_Sort_edge_by_time(edges)
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


myN = 5
myT = 5
gtype = "ER"
# 文件夹路径
my_folder_path = f'../graph_data/graphs_n{myN}_t{myT}/{gtype}'

all_question_types = [
    "When link",
    "When connect",
    "When triadic closure",
    "What neighbors at time",
    "What neighbors in periods",
    "Check triadic closure",
    "Check temporal path",
    "Find temporal path",
    "Sort edge by time"
]

myInstruction = "In an undirected dynamic graph, (u, v, t) means that node u and node v are linked with an undirected edge at time t."

process_files_in_folder(my_folder_path, myN, myT, all_question_types, myInstruction)
