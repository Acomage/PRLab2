import os

mathlib_path = os.path.join(
    os.path.dirname(__file__), "../../data/mathlib4/mathlib4/Mathlib/"
)
mathlib_father_path = os.path.join(
    os.path.dirname(__file__), "../../data/mathlib4/mathlib4/"
)
save_json_path = os.path.join(os.path.dirname(__file__), "../../data/Json/graph.json")
pyvis_output_path = "solve/data/pyvis_output/graph.html"

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


subject_offset = {
    "Algebra": (1.5, -0.5),
    "AlgebraicGeometry": (0.715, 0.0),
    "AlgebraicTopology": (0.675, -1.0),
    "Analysis": (1.5, 0.5),
    "CategoryTheory": (0.5, -1.5),
    "Combinatorics": (0.84, -0.65),
    "Computability": (1.3, 0.19),
    "Condensed": (-0.12, -1.4),
    "Control": (1.32, 0.0),
    "Data": (0.0, 0.0),
    "Dynamics": (0.75, 1.175),
    "FieldTheory": (0.68, -0.62),
    "Geometry": (0.5, 1.0),
    "GroupTheory": (0.5, 1.5),
    "InformationTheory": (1.31, 0.63),
    "LinearAlgebra": (-0.5, 1.5),
    "Logic": (-0.46, -0.64),
    "MeasureTheory": (0.65, 0.635),
    "ModelTheory": (-0.4, 0.45),
    "NumberTheory": (-0.46, 0.35),
    "Order": (-0.5, -1.5),
    "Probability": (0.44, 0.755),
    "RepresentationTheory": (0.0, -1.0),
    "RingTheory": (-1.5, 0.5),
    "SetTheory": (-0.86, 0.0),
    "Topology": (-1.5, -0.5),
}

subject_offset = {subject: (x / 2, y / 2) for subject, (x, y) in subject_offset.items()}

if __name__ == "__main__":
    print(subject_offset)
