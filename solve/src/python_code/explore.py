from graph import Graph
from config import subjects
from typing import Tuple, Dict
import numpy as np
import networkx as nx
from networkx.algorithms.community import modularity


def compute_subjects_data() -> (
    Tuple[
        Dict[str, int],
        Dict[str, int],
        Dict[str, int],
        Dict[str, float],
        Dict[str, float],
    ]
):
    G = Graph.load_from_json()
    subjects_node_num = {subject: 0 for subject in subjects}
    subjects_edge_num = {subject: 0 for subject in subjects}
    subjects_edge_in_subject = {subject: 0 for subject in subjects}
    for node, children in G.nodes.items():
        subject = node.split(".")[1]
        subjects_node_num[subject] += 1
        for child in children:
            subjects_edge_num[subject] += 1
            child_subject = child.split(".")[1]
            subjects_edge_num[child_subject] += 1
            if child_subject == subject:
                subjects_edge_in_subject[subject] += 1
    subject_edge_ratio = {
        subject: subjects_edge_in_subject[subject] / subjects_edge_num[subject]
        for subject in subjects
    }
    node_num = sum(subjects_node_num.values())
    subjects_node_ratio = {
        subject: subjects_node_num[subject] / node_num for subject in subjects
    }
    return (
        subjects_node_num,
        subjects_edge_num,
        subjects_edge_in_subject,
        subject_edge_ratio,
        subjects_node_ratio,
    )


def data_to_latex(
    subjects_node_num: Dict[str, int],
    subjects_edge_num: Dict[str, int],
    subjects_edge_in_subject: Dict[str, int],
    subject_edge_ratio: Dict[str, float],
    subjects_node_ratio: Dict[str, float],
) -> str:
    latex = """
\\begin{table}[H]
\\centering
\\begin{tabular}{cccccc}
    \\toprule
    Subject & Nodes & Edges & EdgesInSameSubject & EdgesRatio & NodesRatio\\\\ 
    \\midrule
"""
    latex += ""
    for subject in subjects:
        latex += f"    {subject} & {subjects_node_num[subject]} & {subjects_edge_num[subject]} & {subjects_edge_in_subject[subject]} & {subject_edge_ratio[subject]:.2f} & {subjects_node_ratio[subject]:.2f}\\\\\n"
    latex += """
    \\bottomrule
\\end{tabular}
\\caption{Number of nodes, edges and edges in the same subject for each subject}
\\label{tab:Ratio}
\\end{table}"""
    return latex


def compute_modularity() -> float:
    G = Graph.load_from_json()
    nodes = np.array(list(G.nodes.keys()))
    subjects = np.array([node.split(".")[1] for node in G.nodes.keys()])
    hash_nodes = {node: i for i, node in enumerate(G.nodes.keys())}
    adjacency_matrix = np.zeros((len(nodes), len(nodes)))
    for node, children in G.nodes.items():
        for child in children:
            adjacency_matrix[hash_nodes[node]][hash_nodes[child]] = 1
            adjacency_matrix[hash_nodes[child]][hash_nodes[node]] = 1

    G_nx = nx.from_numpy_array(adjacency_matrix)
    communities = {}
    for node, label in enumerate(subjects):
        communities.setdefault(label, []).append(node)
    communities = list(communities.values())
    mod = modularity(G_nx, communities)
    return mod


if __name__ == "__main__":
    # print(data_to_latex(*compute_subjects_data()))
    print(compute_modularity())
