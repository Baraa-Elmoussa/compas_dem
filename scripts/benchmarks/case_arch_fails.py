"""Semicircular arch, too thin — the arch that fails.

Same span, rise and voussoir count as ``case_arch_stands.py``, but thickness
is cut to 0.15 m -> thickness/radius = 0.06, below Heyman's ~10.8% minimum
for a semicircular arch under self-weight. No thrust line fits inside such a
thin ring, so the arch is expected to open hinges and collapse under its own
weight, even though friction (mu=0.6) is high enough to rule out sliding as
the failure mode -- this is a genuine geometric (hinging) instability, not a
friction cheat.
"""

from compas_dem.models import BlockModel
from compas_dem.templates import ArchTemplate
from common import assign_stone
from common import build_problem
from common import print_report
from common import run_benchmark

SPAN = 5.0
RISE = 2.5  # == span / 2 -> full semicircular arch, radius = 2.5 m
THICKNESS = 0.15  # thickness / radius = 0.06 << 0.108 (Heyman) -> collapses
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
    problem = build_problem(model, name="benchmark-arch-fails")
    rows = run_benchmark(
        problem,
        displacement_threshold=0.03,
        masonry_dem_kwargs=dict(duration=3.0),
        threedec_kwargs=dict(gravity_steps=10, time=1.0),
    )
    print_report("Arch that fails (t/R = 0.06)", rows)
    return rows


if __name__ == "__main__":
    main()
