import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
import pickle
import csv
import os


def generate_DyER(N, p, T):
    G = nx.erdos_renyi_graph(N, p)
    edges_with_timestamps = [(u, v, random.randint(0, T - 1)) for u, v in G.edges()]
    return G, edges_with_timestamps


def generate_DySB(N, p_in, p_out, T):
    sizes = [N // 2, N - N // 2]
    probs = [[p_in, p_out], [p_out, p_in]]
    G = nx.stochastic_block_model(sizes, probs)
    edges_with_timestamps = [(u, v, random.randint(0, T - 1)) for u, v in G.edges()]
    return G, edges_with_timestamps


def generate_DyFF(N, fwd_prob, bkwd_prob, T):
    G = nx.DiGraph()
    nodes = list(range(N))
    G.add_node(nodes.pop(0))
    for node in nodes:
        targets = set()
        stack = [random.choice(list(G.nodes))]
        while stack:
            current = stack.pop()
            if random.random() < fwd_prob:
                if current not in targets:
                    targets.add(current)
                neighbors = list(G.neighbors(current))
                if neighbors:
                    stack.append((random.choice(neighbors)))
            for target in targets:
                G.add_edge(node, target)
                if random.random() < bkwd_prob:
                    G.add_edge(target, node)
    edges_with_timestamps = [(u, v, random.randint(0, T - 1)) for u, v in G.edges()]
    for node in range(N):
        if node not in G:
            G.add_node(node)
    return G, edges_with_timestamps


def visualize_DyG(G, edges_with_timestamps, filename):
    pos = nx.spring_layout(G)

    edge_labels = {(u, v): t for u, v, t in edges_with_timestamps}
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_size=700, node_color="lightblue", font_size=10, font_weight="bold")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='green')
    plt.title("Dynamic Graph with Timestamps")
    plt.savefig(filename)  # 保存可视化图片
    plt.close()  # 关闭图形窗口
    # 参数设置


def save_graph_to_file(graphs, filename):
    with open(filename, 'wb') as f:
        pickle.dump(graphs, f)


def save_edges_to_csv(edges, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Source', 'Target', 'Timestamp'])
        for edge in edges:
            writer.writerow(edge)


def generate_and_save_all_graphs():
    N = 5
    T = 10
    num_graphs = 200
    gtype = "ER"
    # Parameters for different models
    p_ER = 0.5
    p_in_SB = 0.3
    p_out_SB = 0.1
    fwd_prob_FF = 0.35
    bkwd_prob_FF = 0.2

    all_graphs = {
        "ER": [],
        "SB": [],
        "FF": []
    }

    # Create directory for saving graphs if not exists
    if not os.path.exists(f"graphs_n{N}_t{T}/{gtype}"):
        os.makedirs(f"graphs_n{N}_t{T}/{gtype}")

    # Create directory for saving images if not exists
    if not os.path.exists(f"images_n{N}_t{T}/{gtype}"):
        os.makedirs(f"images_n{N}_t{T}/{gtype}")

    if "ER" in gtype:
        # Generate and store ER graphs
        for i in range(num_graphs):
            G, edges_with_timestamps = generate_DyER(N, p_ER, T)
            all_graphs["ER"].append((G, edges_with_timestamps))
            visualize_DyG(G, edges_with_timestamps, f"images_n{N}_t{T}/{gtype}/ER_graph_{i}.png")
            save_edges_to_csv(edges_with_timestamps, f"graphs_n{N}_t{T}/{gtype}/ER_graph_{i}.csv")
    if "SB" in gtype:
        # Generate and store SB graphs
        for i in range(num_graphs):
            G, edges_with_timestamps = generate_DySB(N, p_in_SB, p_out_SB, T)
            all_graphs["SB"].append((G, edges_with_timestamps))
            visualize_DyG(G, edges_with_timestamps, f"images_n{N}_t{T}/{gtype}/SB_graph_{i}.png")
            save_edges_to_csv(edges_with_timestamps, f"graphs_n{N}_t{T}/{gtype}/SB_graph_{i}.csv")

    if "FF" in gtype:
        # Generate and store FF graphs
        for i in range(num_graphs):
            G, edges_with_timestamps = generate_DyFF(N, fwd_prob_FF, bkwd_prob_FF, T)
            all_graphs["FF"].append((G, edges_with_timestamps))
            visualize_DyG(G, edges_with_timestamps, f"images_n{N}_t{T}/{gtype}/FF_graph_{i}.png")
            save_edges_to_csv(edges_with_timestamps, f"graphs_n{N}_t{T}/{gtype}/FF_graph_{i}.csv")

    for graph_type in all_graphs:
        for i, graph in enumerate(all_graphs[graph_type]):
            save_graph_to_file(graph, f"graphs_n{N}_t{T}/{gtype}/{graph_type}_graph_{i}.pkl")


if __name__ == "__main__":
    generate_and_save_all_graphs()
