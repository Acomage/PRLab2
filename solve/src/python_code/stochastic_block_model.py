from graph_tool.all import (
    Graph,
    minimize_blockmodel_dl,
    adjacency,
)
import json
from config import save_out_json_path, subject_colors, image_output_path
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from networkx.algorithms.community import modularity
import networkx as nx
from typing import Tuple

with open(save_out_json_path, "r") as file:
    nodes = json.load(file)
G = Graph(nodes, hashed=True)

adjacency_matrix = adjacency(G)
node_hash = G.vp.ids

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

state = minimize_blockmodel_dl(G)
subjects = np.array([node_hash[v].split(".")[1] for v in G.vertices()])
labels = np.array([state.get_blocks()[v] for v in G.vertices()])
print(evaluate_clustering(subjects, labels))
state.draw(vertex_size=0.5, output=image_output_path + "SBM.pdf")
