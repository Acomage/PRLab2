"""
这个文件定义了一个图的类，但实际上后面主要使用的是graph_tool和networkx自带的图类。这个类只是在最开始用于数据预处理时稍微用了一下
"""

from typing import List, Dict, Optional
import json
from config import save_json_path


class Node:
    def __init__(self, path: str, parents: Optional[List[str]] = None):
        self.path = path
        self.parents = parents if parents else []

    def add_parent(self, parent: str):
        self.parents.append(parent)


class Graph:
    def __init__(self, nodes: Dict[str, List[str]] = {}):
        self.nodes: Dict[str, List[str]] = nodes

    def add_node(self, node: Node):
        self.nodes[node.path] = node.parents

    def save_as_json(self):
        with open(save_json_path, "w") as file:
            json.dump(self.nodes, file)

    @staticmethod
    def load_from_json() -> "Graph":
        with open(save_json_path, "r") as file:
            nodes = json.load(file)
        return Graph(nodes)


if __name__ == "__main__":
    G = Graph.load_from_json()
    node_num = 0
    edge_num = 0
    for node, edge in G.nodes.items():
        node_num += 1
        edge_num += len(edge)
    print(f"Node number: {node_num}, Edge number: {edge_num}")
