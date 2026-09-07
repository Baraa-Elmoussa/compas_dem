"""Semicircular arch, comfortably thick — the arch that stands.

Span 5 m, rise 2.5 m (= span/2, a full semicircle, radius 2.5 m), thickness
0.5 m -> thickness/radius = 0.20. Heyman's classical minimum-thickness ratio
for a semicircular arch under self-weight alone is ~10.8%, so this arch sits
well above the collapse threshold and is expected to find an internal thrust
line and stand.
"""

from compas_dem.models import BlockModel
from compas_dem.templates import ArchTemplate
from common import assign_stone
from common import build_problem
from common import print_report
from common import run_benchmark

SPAN = 5.0
RISE = 2.5  # == span / 2 -> full semicircular arch, radius = 2.5 m
THICKNESS = 0.5  # thickness / radius = 0.20 >> 0.108 (Heyman) -> stable
DEPTH = 0.4
N_VOUSSOIRS = 16


def build_model() -> BlockModel:
    template = ArchTemplate(rise=RISE, span=SPAN, thickness=THICKNESS, depth=DEPTH, n=N_VOUSSOIRS)
    model = BlockModel.from_template(template)
    model.compute_contacts(tolerance=0.001)
    assign_stone(model)

    for node in model.graph.nodes_where(degree=1):
        model.graph.node_element(node).is_support = True
    return model


def main() -> list:
    model = build_model()
    problem = build_problem(model, name="benchmark-arch-stands")
    rows = run_benchmark(
        problem,
        displacement_threshold=0.03,
        masonry_dem_kwargs=dict(duration=2.0),
        threedec_kwargs=dict(gravity_steps=10, time=1.0),
    )
    print_report("Arch that stands (t/R = 0.20)", rows)
    return rows


if __name__ == "__main__":
    main()
