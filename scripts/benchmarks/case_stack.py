"""Stack of dry-stacked blocks under gravity — a minimal DEM sanity check.

Six identical blocks stacked directly on top of one another with no
eccentricity. With ordinary stone friction (mu=0.6) this configuration is
trivially stable, so it is the "does the basic pipeline work" baseline that
the arch/vault cases are compared against.
"""

from compas.geometry import Box
from compas_dem.models import BlockModel
from common import assign_stone
from common import build_problem
from common import print_report
from common import run_benchmark

N_BLOCKS = 6
BLOCK_SIZE = 0.4  # m


def build_model() -> BlockModel:
    base = Box.from_corner_corner_height([0, 0, 0], [BLOCK_SIZE, BLOCK_SIZE, 0], BLOCK_SIZE)
    boxes = []
    for i in range(N_BLOCKS):
        box = base.copy()
        box.translate([0, 0, i * box.zsize])
        boxes.append(box)

    model = BlockModel.from_boxes(boxes)
    model.compute_contacts()
    assign_stone(model)

    bottom = sorted(model.elements(), key=lambda e: e.point.z)[0]
    bottom.is_support = True
    return model


def main() -> list:
    model = build_model()
    problem = build_problem(model, name="benchmark-stack")
    rows = run_benchmark(
        problem,
        displacement_threshold=0.01,
        masonry_dem_kwargs=dict(duration=1.5),
        threedec_kwargs=dict(gravity_steps=10, time=1.0),
    )
    print_report(f"Stack of {N_BLOCKS} blocks (stable baseline)", rows)
    return rows


if __name__ == "__main__":
    main()
