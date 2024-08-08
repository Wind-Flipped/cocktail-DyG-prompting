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


def visualize_DyG(G, edges_with_timestamps, filename):
    pos = nx.spring_layout(G)

    edge_labels = {(u, v): t for u, v, t in edges_with_timestamps}
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_size=700, node_color="lightblue", font_size=10, font_weight="bold")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='green')
    plt.title("Dynamic Graph with Timestamps")
    plt.savefig(filename)
    plt.close()


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
    N = 10
    T = 20

    num_graphs = 500
    gtype = "ER"

    p_ER = 0.5
    all_graphs = {
        "ER": []
    }

    # Create directory for saving graphs if not exists
    if not os.path.exists(f"graphs_n{N}_t{T}_p{p_ER}/{gtype}"):
        os.makedirs(f"graphs_n{N}_t{T}_p{p_ER}/{gtype}")

    # Create directory for saving images if not exists
    if not os.path.exists(f"images_n{N}_t{T}_p{p_ER}/{gtype}"):
        os.makedirs(f"images_n{N}_t{T}_p{p_ER}/{gtype}")

    if "ER" == gtype:
        # Generate and store ER graphs
        for i in range(num_graphs):
            G, edges_with_timestamps = generate_DyER(N, p_ER, T)
            all_graphs["ER"].append((G, edges_with_timestamps))
            visualize_DyG(G, edges_with_timestamps, f"images_n{N}_t{T}_p{p_ER}/{gtype}/ER_graph_{i}.png")
            save_edges_to_csv(edges_with_timestamps, f"graphs_n{N}_t{T}_p{p_ER}/{gtype}/ER_graph_{i}.csv")

    for graph_type in all_graphs:
        for i, graph in enumerate(all_graphs[graph_type]):
            save_graph_to_file(graph, f"graphs_n{N}_t{T}_p{p_ER}/{gtype}/{graph_type}_graph_{i}.pkl")


if __name__ == "__main__":
    generate_and_save_all_graphs()
