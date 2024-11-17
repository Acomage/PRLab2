from typing import List
from config import subjects

def get_dependency(file_path: str) -> List[str]:
    dependencies = []
    with open(file_path, "r") as file:
        for line in file:
            if line.startswith("import"):
                dependency = line.split("import")[1].strip().split(" ")[0]
                if dependency.startswith("Mathlib.") and dependency.split(".")[1] in subjects:
                    dependencies.append(dependency)
    return dependencies

if __name__ == "__main__":
    import os
    mathlib_path = os.path.join(
    os.path.dirname(__file__), "../../data/mathlib4/mathlib4/Mathlib/"
    )
    print(get_dependency(os.path.join(mathlib_path, "Algebra/Algebra/Basic.lean")))
    