from graph_tool.all import (
    Graph,
    graph_draw,
    minimize_blockmodel_dl,
    minimize_nested_blockmodel_dl,
)
import json
from config import save_out_json_path, subject_colors, image_output_path

with open(save_out_json_path, "r") as file:
    nodes = json.load(file)
G = Graph(nodes, hashed=True)

node_hash = G.vp.ids

vcolor = G.new_vp("string")
for v in G.vertices():
    vcolor[v] = subject_colors[node_hash[v].split(".")[1]]

ecolor = G.new_ep("string")
for v in G.vertices():
    for e in v.out_edges():
        ecolor[e] = vcolor[v]

# def draw_graph():
#     g = graph_draw(
#         G,
#         vertex_fill_color=vcolor,
#         edge_color=ecolor,
#         vertex_size=0.5,
#         output_size=(1000, 1000),
#         output=image_output_path + "graph.pdf"
#     )

state = minimize_blockmodel_dl(G)
state.draw(vertex_size=0.5, output=image_output_path + "SBM.pdf")
