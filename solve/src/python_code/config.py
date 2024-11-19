"""
这个文件是配置文件，用于配置一些共用的路径和颜色
"""

import os

mathlib_path = os.path.join(
    os.path.dirname(__file__), "../../data/mathlib4/mathlib4/Mathlib/"
)
mathlib_father_path = os.path.join(
    os.path.dirname(__file__), "../../data/mathlib4/mathlib4/"
)
save_json_path = os.path.join(os.path.dirname(__file__), "../../data/Json/graph.json")
save_out_json_path = os.path.join(
    os.path.dirname(__file__), "../../data/Json/graph_out.json"
)
pyvis_output_path = "solve/data/pyvis_output/graph.html"
image_output_path = os.path.join(os.path.dirname(__file__), "../../img/")

spectral_cluster_data_path = os.path.join(
    os.path.dirname(__file__), "../../data/Json/spectral_cluster.json"
)

subjects = [
    "Algebra",
    "AlgebraicGeometry",
    "AlgebraicTopology",
    "Analysis",
    "CategoryTheory",
    "Combinatorics",
    "Computability",
    "Condensed",
    "Control",
    "Data",
    "Dynamics",
    "FieldTheory",
    "Geometry",
    "GroupTheory",
    "InformationTheory",
    "LinearAlgebra",
    "Logic",
    "MeasureTheory",
    "ModelTheory",
    "NumberTheory",
    "Order",
    "Probability",
    "RepresentationTheory",
    "RingTheory",
    "SetTheory",
    "Topology",
]

subject_colors = {
    "Algebra": "#1f77b4",
    "AlgebraicGeometry": "#ff7f0e",
    "AlgebraicTopology": "#2ca02c",
    "Analysis": "#46b728",
    "CategoryTheory": "#9467bd",
    "Combinatorics": "#fc76fb",
    "Computability": "#e377c2",
    "Condensed": "#7f7f7f",
    "Control": "#bcbd22",
    "Data": "#17becf",
    "Dynamics": "#aec7e8",
    "FieldTheory": "#ffbb78",
    "Geometry": "#98df8a",
    "GroupTheory": "#ff9896",
    "InformationTheory": "#c5b0d5",
    "LinearAlgebra": "#9c9c94",
    "Logic": "#ffb6b2",
    "MeasureTheory": "#c7c7c7",
    "ModelTheory": "#dbdb8d",
    "NumberTheory": "#9edae5",
    "Order": "#393b79",
    "Probability": "#637939",
    "RepresentationTheory": "#4ced51",
    "RingTheory": "#f7b6df",
    "SetTheory": "#7b4173",
    "Topology": "#96dff7",
}

group_colors = {i: subject_colors[subject] for i, subject in enumerate(subjects)}
