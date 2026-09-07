"""Compute the equilibrium of an arch structure using the CRA method.

To run this script, install `compas_dem` and its dependencies using the preconfigured
"dem-dev" environment in the `compas_dem` repo.

    $ conda env create -f environment.yml
    $ conda activate dem-dev

To generate the input file, run the `dem_dome.py` script first.

"""

import pathlib

from compas_dem.material import Stone
from compas_dem.models import BlockModel
from compas_dem.templates import ArchTemplate
from compas_dem.viewer import DEMViewer

from masonry_dem.core import DiscreteElementModel
from masonry_dem.core.damping import LocalDamping

import compas
from compas_dem.material import Stone
from compas_dem.models import BlockModel
from compas_dem.problem import Problem
from compas_dem.problem import Solver
from compas_dem.viewer import DEMViewer


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
# Material
# =============================================================================

stone: Stone = Stone.from_predefined_material("LimeStone")
stone.density = 2000  # Overwrite default density for limestone
model.add_material(stone)
model.assign_material(stone, elements=list(model.elements()))

# =============================================================================
# Problem setup and solve
# =============================================================================

dem_model = DiscreteElementModel(model)

dem_model.set_global_damping(LocalDamping(lam=0.7))
dem_model.set_joint_stiffness(7e9, 3.5e9)
dem_model.set_contact_law(mu=0.6)
dem_model.setup_problem()

# model.apply_forces([(0, [600000, 0, 0, 0, 0, 0]) ])
# =============================================================================
# Solvers
# =============================================================================

dem_model.solve(n_steps=50000, update_every=5, verbose_every=1000)

# =============================================================================
# Output
# =============================================================================

results = dem_model.extract_results()

# =============================================================================
# Viz
# =============================================================================


viewer = DEMViewer(dem_model)
viewer.add_solution(results)
viewer.show()
