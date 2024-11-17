import os
import single_file
import graph
from config import mathlib_path, mathlib_father_path, subjects


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
                    node.add_child(dependency)
                G.add_node(node)

G.save_as_json()
# G.show_graph_pyvis()
