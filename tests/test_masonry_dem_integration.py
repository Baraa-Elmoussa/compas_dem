from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import pytest
from compas.datastructures import Mesh
from compas_dem.material.generic import GenericMaterial
from compas_dem.models import BlockModel
from compas_dem.problem import Problem
from compas_dem.problem import Solver
from compas_dem.problem.results import Results
from compas_dem.templates import ArchTemplate

pytest.importorskip("masonry_dem", reason="Install masonry_dem to test the MasonryDEM integration.")

from compas_dem.analysis.masonry_dem import _apply_settings  # noqa: E402
from compas_dem.analysis.masonry_dem import _apply_time_series_loads  # noqa: E402
from compas_dem.analysis.masonry_dem import _check_model  # noqa: E402
from compas_dem.analysis.masonry_dem import _gradual_actuator  # noqa: E402
from compas_dem.analysis.masonry_dem import _gravity  # noqa: E402
from compas_dem.analysis.masonry_dem import _resolve_damping  # noqa: E402
from compas_dem.analysis.masonry_dem import masonry_dem_solve  # noqa: E402
from compas_dem.analysis.masonry_dem import select_mode  # noqa: E402
from compas_dem.analysis.resolve import resolve_centroidal_loads  # noqa: E402


# =============================================================================
# Fixtures
# =============================================================================


def build_arch(n: int = 10) -> tuple[BlockModel, Problem]:
    """A supported, contact-computed arch with a Mohr-Coulomb problem on it."""
    arch = ArchTemplate(rise=0.5, span=2.0, thickness=0.2, depth=0.2, n=n)
    model = BlockModel.from_template(arch)
    blocks = list(model.elements())
    model.add_supports([blocks[0].graphnode, blocks[-1].graphnode])

    material = GenericMaterial(Ecm=25e9, density=2000, poisson=0.2)
    model.add_material(material)
    model.assign_material(material, elements=blocks)
    model.compute_contacts(tolerance=0.001)

    problem = Problem(model, name="arch")
    problem.add_boundary_condition("gravity").add_gravity()
    problem.set_contact_model("MohrCoulomb", mu=0.6)
    problem.set_joint_model(kn=7e9, kt=3.5e9)
    return model, problem


def loads_for(model: BlockModel, problem: Problem) -> dict:
    return resolve_centroidal_loads(model, problem.boundary_conditions)


class FakeDEMModel:
    """Records the load tables the layer writes, without running a solve."""

    def __init__(self):
        self.forces = {}
        self.moments = {}
        self.wrenches = {}

    def apply_force(self, block_idx, lin_series):
        self.forces[block_idx] = np.asarray(lin_series, dtype=float)

    def apply_moment(self, block_idx, ang_series):
        self.moments[block_idx] = np.asarray(ang_series, dtype=float)

    def apply_forces(self, loads):
        for idx, wrench in loads:
            self.wrenches[idx] = np.asarray(wrench, dtype=float)


# =============================================================================
# Solver configuration and dispatch
# =============================================================================


def test_masonry_dem_solver_configuration_roundtrip():
    solver = Solver.MasonryDEM(mode="gradual", n_increments=25, damping="rayleigh", damping_params={"alpha": 0.5})

    restored = Solver.__from_data__(solver.__data__)

    assert restored.name == "MasonryDEM"
    assert restored.parameters["mode"] == "gradual"
    assert restored.parameters["n_increments"] == 25
    assert restored.parameters["damping"] == "rayleigh"
    assert restored.parameters["damping_params"] == {"alpha": 0.5}


def test_problem_solve_dispatches_to_masonry_dem_adapter():
    mesh = Mesh.from_vertices_and_faces(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
    )
    model = BlockModel()
    node = model.add_block_from_mesh(mesh)
    model.add_support(node)

    problem = Problem(model)
    problem.add_boundary_condition("gravity").add_gravity()
    problem.set_contact_model("MohrCoulomb", mu=0.6)
    problem.set_solver(Solver.MasonryDEM(n_steps=10))

    sentinel = object()
    with patch("compas_dem.analysis.masonry_dem.masonry_dem_solve", return_value=sentinel) as solve:
        result = problem.solve()

    assert result is sentinel
    assert solve.call_args.args == (problem, model)
    assert solve.call_args.kwargs["n_steps"] == 10
    # Problem.solve drops the unset parameters rather than forwarding None.
    assert "duration" not in solve.call_args.kwargs


# =============================================================================
# Mode selection
# =============================================================================


def test_auto_mode_is_static_without_ramped_loads_or_displacements():
    model, problem = build_arch()
    assert select_mode("auto", loads_for(model, problem), {}) == "static"


def test_auto_mode_is_static_for_instantaneous_loads_only():
    model, problem = build_arch()
    problem.add_point_load_at_centroid(5, [0.0, 0.0, -100.0], loading_type="instantaneous", boundary_condition=problem.boundary_conditions[0])
    assert select_mode("auto", loads_for(model, problem), {}) == "static"


def test_auto_mode_is_gradual_for_a_ramped_load():
    model, problem = build_arch()
    problem.add_point_load_at_centroid(5, [0.0, 0.0, -100.0], loading_type="ramp", boundary_condition=problem.boundary_conditions[0])
    assert select_mode("auto", loads_for(model, problem), {}) == "gradual"


def test_auto_mode_is_gradual_for_a_prescribed_displacement():
    model, problem = build_arch()
    displacements = {5: {"translation": [0.0, 0.0, -0.001], "rotation": [None, None, None]}}
    assert select_mode("auto", loads_for(model, problem), displacements) == "gradual"


def test_explicit_mode_overrides_the_problem():
    model, problem = build_arch()
    problem.add_point_load_at_centroid(5, [0.0, 0.0, -100.0], loading_type="ramp", boundary_condition=problem.boundary_conditions[0])
    assert select_mode("static", loads_for(model, problem), {}) == "static"
    with pytest.raises(ValueError, match="mode must be"):
        select_mode("incremental", {}, {})


# =============================================================================
# Gradual actuator
# =============================================================================


def test_gradual_actuator_splits_a_ramped_load_into_increments():
    model, problem = build_arch()
    problem.add_point_load_at_centroid(5, [300.0, 0.0, -600.0], loading_type="ramp", boundary_condition=problem.boundary_conditions[0])

    blocks, control, increment = _gradual_actuator(loads_for(model, problem), {}, n_increments=3)

    assert blocks == [5]
    assert control == "force"
    assert increment == pytest.approx([100.0, 0.0, -200.0, 0.0, 0.0, 0.0])


def test_gradual_actuator_prefers_a_prescribed_displacement_over_a_ramped_load():
    model, problem = build_arch()
    problem.add_point_load_at_centroid(5, [300.0, 0.0, 0.0], loading_type="ramp", boundary_condition=problem.boundary_conditions[0])
    displacements = {4: {"translation": [None, None, -0.004], "rotation": [None, None, None]}}

    blocks, control, increment = _gradual_actuator(loads_for(model, problem), displacements, n_increments=4)

    assert blocks == [4]
    assert control == "displacement"
    # Unconstrained components come through as None and are imposed as zero.
    assert increment == pytest.approx([0.0, 0.0, -0.001, 0.0, 0.0, 0.0])


def test_gradual_actuator_rejects_diverging_increments():
    model, problem = build_arch()
    bc = problem.boundary_conditions[0]
    problem.add_point_load_at_centroid(4, [100.0, 0.0, 0.0], loading_type="ramp", boundary_condition=bc)
    problem.add_point_load_at_centroid(5, [200.0, 0.0, 0.0], loading_type="ramp", boundary_condition=bc)

    with pytest.raises(ValueError, match="one shared increment"):
        _gradual_actuator(loads_for(model, problem), {}, n_increments=3)


def test_gradual_actuator_needs_something_to_drive():
    model, problem = build_arch()
    with pytest.raises(ValueError, match="neither"):
        _gradual_actuator(loads_for(model, problem), {}, n_increments=3)


# =============================================================================
# Load time series
# =============================================================================


def test_time_series_combines_ramped_and_instantaneous_loads():
    model, problem = build_arch()
    bc = problem.boundary_conditions[0]
    problem.add_point_load_at_centroid(5, [0.0, 0.0, -300.0], loading_type="ramp", boundary_condition=bc)
    problem.add_point_load_at_centroid(5, [0.0, 0.0, -100.0], loading_type="instantaneous", boundary_condition=bc)

    dem_model = FakeDEMModel()
    _apply_time_series_loads(dem_model, loads_for(model, problem), duration=2.0)

    # Sampled at [0, 0.98T, T] as [i, r + i, r]: the ramp grows and holds while
    # the instantaneous part is applied at t=0 and released at the end.
    expected = [[0.0, 0.0, 0.0, -100.0], [1.96, 0.0, 0.0, -400.0], [2.0, 0.0, 0.0, -300.0]]
    assert np.allclose(dem_model.forces[5], expected)
    assert 5 not in dem_model.moments


def test_time_series_carries_the_moment_of_an_eccentric_load():
    model, problem = build_arch()
    block = model._block(5)
    point = [block.point.x + 0.1, block.point.y, block.point.z]
    problem.add_point_load_at_point(5, point, [0.0, 0.0, -100.0], boundary_condition=problem.boundary_conditions[0])

    dem_model = FakeDEMModel()
    _apply_time_series_loads(dem_model, loads_for(model, problem), duration=1.0)

    # r x F about the centroid: 0.1 in x crossed with -100 in z is +10 about y.
    assert dem_model.moments[5][-1] == pytest.approx([1.0, 0.0, 10.0, 0.0])


def test_time_series_needs_a_positive_duration():
    model, problem = build_arch()
    problem.add_point_load_at_centroid(5, [0.0, 0.0, -100.0], boundary_condition=problem.boundary_conditions[0])
    with pytest.raises(ValueError, match="positive duration"):
        _apply_time_series_loads(FakeDEMModel(), loads_for(model, problem), duration=0.0)


# =============================================================================
# Configuration mapping and validation
# =============================================================================


def test_damping_is_resolved_by_name_with_parameters():
    from masonry_dem.core.damping import LocalDamping

    damping = _resolve_damping("local", {"lam": 0.4})
    assert isinstance(damping, LocalDamping)
    assert damping.lam == 0.4

    instance = LocalDamping(lam=0.3)
    assert _resolve_damping(instance, None) is instance

    with pytest.raises(ValueError, match="not recognised"):
        _resolve_damping("magic", None)


def test_settings_reject_unknown_attributes():
    dem_model = Mock()
    _apply_settings(dem_model, {"safety": 0.2, "tolerance": 1e-6})
    assert dem_model.safety == 0.2
    assert dem_model.tolerance == 1e-6

    with pytest.raises(ValueError, match="Unknown masonry_dem settings"):
        _apply_settings(Mock(), {"not_a_setting": 1})


def test_model_without_contacts_is_rejected():
    arch = ArchTemplate(rise=0.5, span=2.0, thickness=0.2, depth=0.2, n=5)
    model = BlockModel.from_template(arch)
    with pytest.raises(ValueError, match="compute_contacts"):
        _check_model(model)


def test_conflicting_gravity_between_boundary_conditions_is_rejected():
    model, problem = build_arch()
    problem.add_boundary_condition("live").add_gravity(g=1.62)
    with pytest.raises(ValueError, match="different gravitational accelerations"):
        _gravity(problem)


def test_n_steps_and_duration_are_mutually_exclusive():
    model, problem = build_arch()
    with pytest.raises(ValueError, match="not both"):
        masonry_dem_solve(problem, model, n_steps=10, duration=1.0)


def test_static_mode_needs_a_run_length():
    model, problem = build_arch()
    with pytest.raises(ValueError, match="n_steps or duration"):
        masonry_dem_solve(problem, model, mode="static")


def test_prescribed_displacement_on_a_support_is_reported(capsys):
    from compas_dem.analysis.masonry_dem import _warn_about_pinned_displacements

    model, _ = build_arch()
    support = next(model.supports()).graphnode
    _warn_about_pinned_displacements(model, {support: {"translation": [0.0, 0.0, -0.001]}})

    assert f"blocks [{support}] are supports" in capsys.readouterr().out


# =============================================================================
# End-to-end
# =============================================================================


def test_static_gravity_run_returns_results_keyed_to_the_problem():
    model, problem = build_arch()
    problem.set_solver(Solver.MasonryDEM(n_steps=2000, verbose_every=0))

    results = problem.solve()

    assert isinstance(results, Results)
    assert results.model_id == str(model.guid)
    assert results.problem_id == str(problem.guid)
    assert results.metadata["solver"] == "MasonryDEM"
    assert results.metadata["mode"] == "static"
    assert results.metadata["mu"] == 0.6
    assert set(results.nodes()) == {block.graphnode for block in model.elements()}
    # Every joint of a settled arch carries compression.
    assert all(results.force_magnitude(edge) > 0 for edge in results.edges())


def test_gradual_force_control_reaches_the_ramped_load():
    model, problem = build_arch()
    crown = 5
    problem.add_point_load_at_centroid(crown, [200.0, 0.0, 0.0], loading_type="ramp", boundary_condition=problem.boundary_conditions[0])
    problem.set_solver(Solver.MasonryDEM(n_increments=2, max_steps=40000, ramp_time=5e-3, verbose_every=0))

    results = problem.solve()

    increments = results.metadata["increments"]
    assert results.metadata["mode"] == "gradual"
    # One settle record plus one per increment.
    assert [record["control"] for record in increments] == ["settle", "force", "force"]
    assert increments[-1]["reaction"] == pytest.approx([200.0, 0.0, 0.0])
    assert results.metadata["converged"]


def test_gradual_displacement_control_reaches_the_prescribed_movement():
    model, problem = build_arch()
    crown = 5
    problem.add_displacement(crown, [0.0, 0.0, -0.002], boundary_condition=problem.boundary_conditions[0])
    problem.set_solver(Solver.MasonryDEM(n_increments=2, max_steps=40000, ramp_time=5e-3, verbose_every=0))

    results = problem.solve()

    increments = results.metadata["increments"]
    assert [record["control"] for record in increments] == ["settle", "displacement", "displacement"]
    assert increments[-1]["displacement"] == pytest.approx([0.0, 0.0, -0.002], abs=1e-5)
    # The actuator has to push against the arch to impose the settlement.
    assert np.linalg.norm(increments[-1]["reaction"]) > np.linalg.norm(increments[0]["reaction"])


def test_results_survive_a_json_roundtrip():
    import compas

    model, problem = build_arch()
    problem.set_solver(Solver.MasonryDEM(n_steps=500, verbose_every=0))

    results = problem.solve()
    restored = compas.json_loads(compas.json_dumps(results))

    assert restored.model_id == results.model_id
    assert restored.metadata["solver"] == "MasonryDEM"
    assert set(restored.edges()) == set(results.edges())
