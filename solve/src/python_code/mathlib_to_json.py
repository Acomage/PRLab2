"""
这个文件的作用是将mathlib中的所有文件的依赖关系保存为json文件，依赖关系以入邻接表和出邻接表的形式保存。
"""

import os
import single_file
import graph
from config import mathlib_path, mathlib_father_path, subjects, save_out_json_path
import json


G = graph.Graph()

for subject in subjects:
    for root, dirs, files in os.walk(os.path.join(mathlib_path, subject)):
        for file in files:
            if file.endswith(".lean"):
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, mathlib_father_path)
                dotted_path = relative_path.replace("/", ".")
                dependencies = single_file.get_dependency(full_path)
                node = graph.Node(dotted_path[:-5])
                for dependency in dependencies:
                    node.add_parent(dependency)
                G.add_node(node)

G.save_as_json()

G = graph.Graph.load_from_json()
out_json = {node: [] for node in G.nodes}
for node, children in G.nodes.items():
    for child in children:
        out_json[child].append(node)

with open(save_out_json_path, "w") as file:
    json.dump(out_json, file)
