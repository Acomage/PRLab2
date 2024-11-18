from typing import List, Dict
import json
from config import (
    save_json_path,
    pyvis_output_path,
    subject_colors,
    subjects,
    subject_offset,
)
from matplotlib.lines import Line2D


class Node:
    def __init__(self, path: str, children: List[str] = None):
        self.path = path
        self.children = children if children else []

    def add_child(self, child: str):
        self.children.append(child)


class Graph:
    def __init__(self, nodes: Dict[str, List[str]] = {}):
        self.nodes: Dict[str, List[str]] = nodes

    def add_node(self, node):
        self.nodes[node.path] = node.children

    def save_as_json(self):
        with open(save_json_path, "w") as file:
            json.dump(self.nodes, file)

    @staticmethod
    def load_from_json() -> "Graph":
        with open(save_json_path, "r") as file:
            nodes = json.load(file)
        return Graph(nodes)

    @staticmethod
    def random_load_from_json(random_seed: int, probability) -> "Graph":
        """Randomly select some nodes from the json file to show so that the graph is not too large"""
        import random

        random.seed(random_seed)
        with open(save_json_path, "r") as file:
            nodes = json.load(file)
        new_nodes = {}
        for node in nodes:
            if random.random() < probability:
                new_nodes[node] = nodes[node]
        return Graph(new_nodes)

    def show_graph_networkx(self):
        import networkx as nx
        import matplotlib.pyplot as plt

        G = nx.Graph()
        for node in self.nodes:
            subject = node.split(".")[1]
            G.add_node(node, group=subject)
        for node in self.nodes:
            for child in self.nodes[node]:
                if child in self.nodes:
                    G.add_edge(node, child)
        group_colors = {subject: subject_colors[subject] for subject in subjects}
        groups = nx.get_node_attributes(G, "group")
        pos = nx.spring_layout(G, iterations=50, seed=42, k=0.2)
        offset = {subject: subject_offset[subject] for subject in subjects}
        for node, (x, y) in pos.items():
            group = groups[node]
            ox, oy = offset[group]
            pos[node] = (x + ox, y + oy)
        colors = [group_colors[groups[node]] for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=0.2)
        edges_color = [group_colors[groups[edge[1]]] for edge in G.edges]
        nx.draw_networkx_edges(G, pos, edge_color=edges_color, width=0.1)
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=subject,
                markersize=10,
                markerfacecolor=color,
            )
            for subject, color in group_colors.items()
        ]
        plt.legend(handles=legend_elements, loc="upper right", fontsize="xx-small")

        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    G = Graph.load_from_json()
    node_num = 0
    edge_num = 0
    for node, edge in G.nodes.items():
        node_num += 1
        edge_num += len(edge)
    print(f"Node number: {node_num}, Edge number: {edge_num}")
