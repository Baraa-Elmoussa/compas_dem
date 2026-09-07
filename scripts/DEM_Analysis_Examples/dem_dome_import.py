"""Compute the equilibrium of an arch structure using the CRA method.

To run this script, install `compas_dem` and its dependencies using the preconfigured
"dem-dev" environment in the `compas_dem` repo.

    $ conda env create -f environment.yml
    $ conda activate dem-dev

To generate the input file, run the `dem_dome.py` script first.

"""

import pathlib

import compas
from compas_dem.material import Stone
from compas_dem.models import BlockModel
from compas_dem.problem import Problem
from compas_dem.problem import Solver
from compas_dem.viewer import DEMViewer
import compas.geometry as cg
from compas.datastructures import Mesh
from compas_viewer.viewer import Viewer
# =============================================================================
# Import
# =============================================================================


file = "/Users/belmoussa/My_Files/Other_libs/Applications/meshes.json"
meshes = compas.json_load(file)
model = BlockModel()

for mesh in meshes:
    model.add_block_from_mesh(mesh)
for block in model.elements():
    if block.point[2] < 0.4:
        block.is_support = True

# =============================================================================
# Output
# =============================================================================

Here = pathlib.Path(__file__).parent
compas.json_dump(model, Here / "half_dome_saint_bartalo.json")
model.compute_contacts()
viewer = DEMViewer(model)
viewer.setup()
viewer.config.renderer.show_grid = False
viewer.show()
raise
# =============================================================================
# Material
# =============================================================================

stone: Stone = Stone.from_predefined_material("LimeStone")
stone.density = 2000  # Overwrite default density for limestone
model.add_material(stone)
model.assign_material(stone, elements=list(model.elements()))

# =============================================================================
# Problem setup and solve
# =============================================================================

problem = Problem(model)
problem.set_contact_model("MohrCoulomb", mu=0.5, c=0.0)
lmgc90_solver = Solver.LMGC90(duration=0.2, n_steps=1000, urf_threshold=1e-3, theta=0.5)
rbe_solver = Solver.PRD(solver="CLARABEL")
tdec_solver = Solver.ThreeDEC()
# problem.set_solver(tdec_solver)
# solution_3dec = problem.solve()

problem.set_solver(lmgc90_solver)
solution_lmgc90 = problem.solve()
# =============================================================================
# Viz
# =============================================================================

viewer = DEMViewer(model)
# viewer.add_solution(solution_3dec, name="3DEC")
viewer.add_solution(solution_lmgc90, name="LMGC90")
viewer.show()
