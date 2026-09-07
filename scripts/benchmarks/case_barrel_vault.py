"""Barrel vault under self-weight -- a semicircular arch profile extruded.

Span 4 m, rise 0.8 m -> transverse radius 2.9 m; thickness 0.5 m gives
thickness/radius = 0.17, comfortably above the arch collapse threshold, so
each transverse voussoir strip is expected to stand. 7x5 voussoirs (38
blocks total, boundary rings flagged as supports by the template itself).
"""

from compas_dem.models import BlockModel
from compas_dem.templates import BarrelVaultTemplate
from common import assign_stone
from common import build_problem
from common import print_report
from common import run_benchmark

SPAN = 4.0
LENGTH = 2.5
RISE = 0.8
THICKNESS = 0.5
VOU_SPAN = 7
VOU_LENGTH = 5


def build_model() -> BlockModel:
    template = BarrelVaultTemplate(
        span=SPAN,
        length=LENGTH,
        rise=RISE,
        thickness=THICKNESS,
        vou_span=VOU_SPAN,
        vou_length=VOU_LENGTH,
    )
    model = BlockModel.from_barrelvault(template)
    model.compute_contacts(tolerance=0.001)
    assign_stone(model)
    return model


def main() -> list:
    model = build_model()
    problem = build_problem(model, name="benchmark-barrel-vault")
    rows = run_benchmark(
        problem,
        displacement_threshold=0.03,
        masonry_dem_kwargs=dict(duration=2.0),
        threedec_kwargs=dict(gravity_steps=10, time=1.0, timeout=600),
    )
    n_blocks = sum(1 for _ in model.elements())
    print_report(f"Barrel vault ({n_blocks} blocks)", rows)
    return rows


if __name__ == "__main__":
    main()
