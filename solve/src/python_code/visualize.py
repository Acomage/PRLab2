"""
这个文件用来可视化，包括使用学科分类对图着色和使用谱聚类的结果对图着色
"""

from graph_tool.all import (
    Graph,
    graph_draw,
)
import json
from config import (
    save_out_json_path,
    subject_colors,
    image_output_path,
    spectral_cluster_data_path,
    group_colors,
)

with open(save_out_json_path, "r") as file:
    nodes = json.load(file)
G = Graph(nodes, hashed=True)

node_hash = G.vp.ids

vertex_origin_color = G.new_vp("string")
for v in G.vertices():
    vertex_origin_color[v] = subject_colors[node_hash[v].split(".")[1]]

edge_origin_color = G.new_ep("string")
for v in G.vertices():
    for e in v.out_edges():
        edge_origin_color[e] = vertex_origin_color[v]


def draw_origin():
    graph_draw(
        G,
        vertex_fill_color=vertex_origin_color,
        edge_color=edge_origin_color,
        vertex_size=0.5,
        output_size=(1000, 1000),
        output=image_output_path + "graph.pdf",
    )


with open(spectral_cluster_data_path, "r") as file:
    nodes_with_spectral_cluster_lable = json.load(file)

vertex_spectrum_color = G.new_vp("string")
for v in G.vertices():
    vertex_spectrum_color[v] = group_colors[
        nodes_with_spectral_cluster_lable[node_hash[v]]
    ]

edge_spectrum_color = G.new_ep("string")
for v in G.vertices():
    for e in v.out_edges():
        edge_spectrum_color[e] = vertex_spectrum_color[v]


def draw_spectral_cluster():
    graph_draw(
        G,
        vertex_fill_color=vertex_spectrum_color,
        edge_color=edge_spectrum_color,
        vertex_size=0.5,
        output_size=(1000, 1000),
        output=image_output_path + "graph_spectral_cluster.pdf",
    )


if __name__ == "__main__":
    draw_origin()
    draw_spectral_cluster()
