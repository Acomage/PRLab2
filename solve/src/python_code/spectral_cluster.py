"""
这个文件用于对图进行谱聚类，并且评估聚类效果
"""

from typing import Tuple
from graph import Graph
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import networkx as nx
from networkx.algorithms.community import modularity
from config import spectral_cluster_data_path
import json

G = Graph.load_from_json()
nodes = np.array(list(G.nodes.keys()))
subjects = np.array([node.split(".")[1] for node in G.nodes.keys()])
hash_nodes = {node: i for i, node in enumerate(G.nodes.keys())}
adjacency_matrix = np.zeros((len(nodes), len(nodes)))
for node, parents in G.nodes.items():
    for parent in parents:
        adjacency_matrix[hash_nodes[node]][hash_nodes[parent]] = 1
        adjacency_matrix[hash_nodes[parent]][hash_nodes[node]] = 1


def evaluate_clustering(
    subjects: np.ndarray, labels: np.ndarray
) -> Tuple[float, float, float]:
    ari = adjusted_rand_score(subjects, labels)
    nmi = normalized_mutual_info_score(subjects, labels)
    G_nx = nx.from_numpy_array(adjacency_matrix)
    communities = {}
    for node, label in enumerate(labels):
        communities.setdefault(label, []).append(node)
    communities = list(communities.values())
    mod = modularity(G_nx, communities)
    return ari, nmi, mod


if __name__ == "__main__":
    n_clusters = 26
    spectral = SpectralClustering(
        n_clusters=n_clusters, affinity="precomputed", assign_labels="kmeans"
    )
    labels = np.array(spectral.fit_predict(adjacency_matrix))
    ari, nmi, mod = evaluate_clustering(subjects, labels)
    print(f"ARI: {ari}, NMI: {nmi}, Modularity: {mod}")
    output_data = {node: int(label) for node, label in zip(nodes, labels)}
    with open(spectral_cluster_data_path, "w") as file:
        json.dump(output_data, file)
