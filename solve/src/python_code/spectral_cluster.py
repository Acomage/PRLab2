from graph import Graph
import numpy as np
from sklearn.cluster import SpectralClustering

G = Graph.load_from_json()
nodes = np.array(list(G.nodes.keys()))
subjects = np.array([node.split(".")[1] for node in G.nodes.keys()])
hash_nodes = {node: i for i, node in enumerate(G.nodes.keys())}
adjacency_matrix = np.zeros((len(nodes), len(nodes)))
for node, children in G.nodes.items():
    for child in children:
        adjacency_matrix[hash_nodes[node]][hash_nodes[child]] = 1
        adjacency_matrix[hash_nodes[child]][hash_nodes[node]] = 1

if __name__ == "__main__":
    n_clusters = 10
    spectral = SpectralClustering(
        n_clusters=n_clusters, affinity="precomputed", assign_labels="kmeans"
    )
    labels = spectral.fit_predict(adjacency_matrix)
    print(labels)
