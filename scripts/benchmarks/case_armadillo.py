"""Armadillo block decomposition under gravity -- a large, irregular stress test.

Not a classical masonry typology: 399 free-form blocks decomposed from the
Stanford Armadillo mesh (``data/armadillo.obj``), the same geometry used in
``docs/examples/dem_armadillo.py``. Any block with a vertex near the ground
(z < 0.5) is a support, mirroring that example, so the statue effectively
rests on a floor rather than springing from a handful of piers.

This case exists to stress-test the two backends on a large, irregular
contact graph -- far more blocks and far less regular contact geometry than
the arches/vaults -- rather than to make a structural-stability point.
"""

import argparse
import pathlib

from compas.datastructures import Mesh
from compas.files import OBJ
from compas_dem.models import BlockModel
from common import assign_stone
from common import build_problem
from common import print_report
from common import run_benchmark

REPO_ROOT = pathlib.Path(__file__).parent.parent.parent
DATA_FILE = REPO_ROOT / "data" / "armadillo.obj"
GROUND_Z = 0.5


def build_model() -> BlockModel:
    obj = OBJ(DATA_FILE)
    obj.read()

    meshes = []
    for name in obj.objects:  # type: ignore
        vertices, faces = obj.objects[name]  # type: ignore
        mesh = Mesh.from_vertices_and_faces(vertices, faces)
        mesh.translate([-10, -10, 0])
        mesh.name = name
        meshes.append(mesh)

    model = BlockModel.from_boxes(meshes)
    model.compute_contacts(tolerance=0.1)
    assign_stone(model)

    for element in model.elements():
        if any(element.modelgeometry.vertex_attribute(v, "z") < GROUND_Z for v in element.modelgeometry.vertices()):
            element.is_support = True
    return model


def _show_inspection(problem) -> None:
    """Pop a viewer showing block volumes and supports (red) before solving."""
    problem.inspect_model(show_blocks=True, face_indices=False, show_loads=False, show_supports=True, kill=False)


def _show_solution(model, rows) -> None:
    """Pop a viewer overlaying every backend's solved (post-gravity) shape."""
    from compas_dem.viewer import DEMViewer

    viewer = DEMViewer(model)
    viewer.setup()
    for row in rows:
        if row.status == "ok" and row.results is not None:
            viewer.add_solution(row.results, name=row.solver, scale=1.0)
    viewer.show()


def main(inspect: bool = False, view: bool = False) -> list:
    model = build_model()
    problem = build_problem(model, name="benchmark-armadillo")

    if inspect:
        _show_inspection(problem)
        return []

    rows = run_benchmark(
        problem,
        displacement_threshold=0.03,
        masonry_dem_kwargs=dict(duration=0.3),
        threedec_kwargs=dict(gravity_steps=10, time=1.0, timeout=1200),
    )
    n_blocks = sum(1 for _ in model.elements())
    n_supports = sum(1 for e in model.elements() if e.is_support)
    print_report(f"Armadillo ({n_blocks} blocks, {n_supports} supports)", rows)

    if view:
        _show_solution(model, rows)

    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect", action="store_true", help="Show block geometry and supports (red) in a viewer, then exit without solving.")
    parser.add_argument("--view", action="store_true", help="After solving, open a viewer overlaying each backend's solved shape.")
    args = parser.parse_args()
    main(inspect=args.inspect, view=args.view)
