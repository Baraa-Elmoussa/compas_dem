"""Cross (groin) vault under self-weight, built from curated stereotomy data.

Reuses the same digitized cross-vault geometry as
``scripts/DEM_Analysis_Examples/dem_vault_cross.py`` (184 blocks): real
voussoir cutting rather than a parametric approximation, so this is the
heaviest and most geometrically realistic case in the suite. Perimeter
blocks that only touch one neighbour (the springers resting on the
abutments) are flagged as supports.
"""

import pathlib

from compas.datastructures import Mesh
from compas.files import OBJ
from compas_dem.models import BlockModel
from common import assign_stone
from common import build_problem
from common import print_report
from common import run_benchmark

REPO_ROOT = pathlib.Path(__file__).parent.parent.parent
DATA_FILE = REPO_ROOT / "data" / "crossvault.obj"
SCALE = 0.025


def build_model() -> BlockModel:
    obj = OBJ(DATA_FILE)
    obj.read()

    meshes = []
    for name in obj.objects:  # type: ignore
        vertices, faces = obj.objects[name]  # type: ignore
        mesh = Mesh.from_vertices_and_faces(vertices, faces)
        mesh.scale(SCALE, SCALE, SCALE)
        mesh.name = name
        meshes.append(mesh)

    model = BlockModel.from_boxes(meshes)
    model.compute_contacts(tolerance=0.001)
    assign_stone(model)

    for element in model.elements():
        if model.graph.degree(element.graphnode) == 1:
            element.is_support = True
    return model


def main() -> list:
    model = build_model()
    problem = build_problem(model, name="benchmark-cross-vault")
    rows = run_benchmark(
        problem,
        displacement_threshold=0.03,
        masonry_dem_kwargs=dict(duration=1.5),
        threedec_kwargs=dict(gravity_steps=10, time=1.0, timeout=900),
    )
    n_blocks = sum(1 for _ in model.elements())
    print_report(f"Cross vault ({n_blocks} blocks)", rows)
    return rows


if __name__ == "__main__":
    main()
