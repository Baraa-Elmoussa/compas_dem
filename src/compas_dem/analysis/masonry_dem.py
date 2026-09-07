from typing import Optional
from typing import Union

import numpy as np

from compas_dem.analysis.resolve import resolve_centroidal_displacements
from compas_dem.analysis.resolve import resolve_centroidal_loads
from compas_dem.models import BlockModel
from compas_dem.problem.problem import Problem
from compas_dem.problem.results import Results

try:
    from masonry_dem.core import DiscreteElementModel
    from masonry_dem.core.damping import AdaptiveGlobalDamping
    from masonry_dem.core.damping import Damping
    from masonry_dem.core.damping import LocalDamping
    from masonry_dem.core.damping import NoDamping
    from masonry_dem.core.damping import RayleighDamping
    from masonry_dem.core.damping import ViscousDamping
except (ImportError, FileNotFoundError):
    raise ImportError("masonry_dem is not installed. Install it locally to use the MasonryDEM solver.")


#: Damping strategies selectable by name from the solver configuration.
DAMPING_MODELS: dict[str, type] = {
    "local": LocalDamping,
    "viscous": ViscousDamping,
    "rayleigh": RayleighDamping,
    "adaptive": AdaptiveGlobalDamping,
    "none": NoDamping,
}

#: Attributes of ``DiscreteElementModel`` that ``settings`` is allowed to reach.
#: Everything here is consumed by ``setup_problem()`` or the solver loop, so the
#: escape hatch stays a documented surface rather than arbitrary ``setattr``.
_SETTINGS_WHITELIST = frozenset(
    (
        "beta",
        "safety",
        "tolerance",
        "check_interval",
        "warnings",
        "use_flat_forces",
        "use_njit",
        "rigid_body_euler",
        "adaptive_time_step",
        "dt_FOS_critical",
        "contact_method",
        "cp_solver",
        "max_fcp_iters",
        "CTOL",
        "contact_stale_steps",
        "nominal_contact_width",
        "slip_retol",
        "rot_retol",
        "hist_radius",
        "ff_centroid_point",
        "merge_joined_interfaces",
        "save_joined",
        "cundall_max_iter",
        "cundall_tol",
        "cundall_angle_max",
        "cundall_angle_min",
        "cundall_neighbor",
        "cundall_adist",
        "cundall_deep_recheck",
    )
)


def masonry_dem_solve(
    problem: Problem,
    model: BlockModel,
    mode: str = "auto",
    n_steps: Optional[int] = None,
    duration: Optional[float] = None,
    damping: Union[str, "Damping"] = "local",
    damping_params: Optional[dict] = None,
    kn: Optional[float] = None,
    kt: Optional[float] = None,
    update_every: int = 10,
    verbose_every: int = 1000,
    convergence_check: bool = True,
    n_increments: int = 10,
    ramp_time: float = 1e-2,
    max_steps: int = 50_000,
    settle_first: bool = True,
    stop_on_failure: bool = True,
    join_blocks: Optional[list[list[int]]] = None,
    hist_every: Optional[int] = None,
    log_ring: Optional[int] = None,
    vtk_every: Optional[int] = None,
    vtk_dir: Optional[str] = None,
    extract_step: int = -1,
    settings: Optional[dict] = None,
) -> Results:
    """Translate a Problem into a ``masonry_dem`` model, run it, and return the results.

    Requires ``model.compute_contacts()`` to have been called first.

    The problem picks the solve mode by itself unless ``mode`` says otherwise:
    a prescribed displacement or a ramped load means the load is applied
    incrementally (``solve_gradual``), anything else is integrated in one run
    (``solve``). See :func:`select_mode`.

    Parameters
    ----------
    problem : :class:`~compas_dem.problem.Problem`
    model : :class:`~compas_dem.models.BlockModel`
    mode : str, optional
        ``"auto"`` (default), ``"static"`` or ``"gradual"``.
    n_steps : int, optional
        Number of time steps for the static run. Mutually exclusive with ``duration``.
    duration : float, optional
        Simulated time for the static run [s]; ``n_steps`` is derived from the
        critical timestep. Also sets the time base of the ramp/instantaneous
        load series in static mode.
    damping : str or :class:`masonry_dem.core.damping.Damping`, optional
        Damping strategy: one of ``"local"``, ``"viscous"``, ``"rayleigh"``,
        ``"adaptive"``, ``"none"``, or a ready-made ``Damping`` instance.
        Default ``"local"``.
    damping_params : dict, optional
        Keyword arguments for the named damping model, e.g. ``{"lam": 0.7}``.
    kn, kt : float, optional
        Joint normal / tangential stiffness [Pa/m]. Fall back to the problem's
        joint model, then to the ``masonry_dem`` defaults.
    update_every : int, optional
        Contact-geometry refresh interval, in steps. Default ``10``.
    verbose_every : int, optional
        Solver progress interval, in steps. ``0`` suppresses it. Default ``1000``.
    convergence_check : bool, optional
        Stop a static run early once the unbalanced force ratio drops below the
        model tolerance. Default ``True``.
    n_increments : int, optional
        Number of increments in gradual mode. Default ``10``.
    ramp_time : float, optional
        Minimum time over which one increment is imposed [s]. Default ``1e-2``.
    max_steps : int, optional
        Step ceiling for each relax phase in gradual mode. Default ``50000``.
    settle_first : bool, optional
        Relax under self-weight before the first increment. Default ``True``.
    stop_on_failure : bool, optional
        Stop the incremental run at the first non-converged increment. Default ``True``.
    join_blocks : list[list[int]], optional
        Groups of block indices to weld into single rigid bodies.
    hist_every, log_ring, vtk_every, vtk_dir
        Recording configuration, forwarded to ``DiscreteElementModel.log``.
    extract_step : int, optional
        Step to extract the results from. Default ``-1`` (last).
    settings : dict, optional
        Escape hatch for the remaining ``DiscreteElementModel`` attributes
        (``safety``, ``tolerance``, ``contact_method``, the Cundall parameters, ...),
        applied before ``setup_problem()``.

    Returns
    -------
    :class:`~compas_dem.problem.Results`

    Raises
    ------
    ValueError
        If the model has no contacts, if its block indices are not the graph node
        indices, or if the problem cannot be expressed in the selected mode.
    """
    if n_steps is not None and duration is not None:
        raise ValueError("Provide either n_steps or duration for the static run, not both.")

    _check_model(model)

    dem_model = DiscreteElementModel(model)

    # ------------------------------------------------------------------
    # Configuration — everything here has to precede setup_problem()
    # ------------------------------------------------------------------
    contact_model = problem.contact_properties.contact_model
    dem_model.set_contact_law(
        mu=contact_model.mu,
        cohesion=contact_model.c or 0.0,
        tension_cutoff=contact_model.t_c or 0.0,
    )

    joint_model = problem.contact_properties.joint_model
    k_n = kn if kn is not None else getattr(joint_model, "kn", None)
    k_t = kt if kt is not None else getattr(joint_model, "kt", None)
    if k_n is not None:
        dem_model.set_joint_stiffness(k_n, k_t)
    elif k_t is not None:
        raise ValueError("A tangential joint stiffness was given without a normal one. Set both, via problem.set_joint_model(kn=..., kt=...) or the solver's kn/kt.")

    dem_model.set_global_damping(_resolve_damping(damping, damping_params))
    dem_model.gravity = np.array([0.0, 0.0, -_gravity(problem)])

    for group in join_blocks or []:
        dem_model.join_blocks(group)

    _apply_settings(dem_model, settings)

    # log() leaves every setting it is not given untouched.
    dem_model.log(ring=log_ring, hist_every=hist_every, vtk_every=vtk_every, vtk_dir=vtk_dir)

    # The critical timestep is only known once the geometry has been processed,
    # and the load time series are written against it, so loads come after setup.
    dem_model.setup_problem()

    # ------------------------------------------------------------------
    # Loads and boundary conditions
    # ------------------------------------------------------------------
    loads = resolve_centroidal_loads(model, problem.boundary_conditions)
    displacements = resolve_centroidal_displacements(problem.boundary_conditions)
    _warn_about_pinned_displacements(model, displacements)

    mode = select_mode(mode, loads, displacements)

    if mode == "static":
        if n_steps is None and duration is None:
            raise ValueError("A static masonry_dem run needs either n_steps or duration.")
        # The load series are written against the run itself, so a step count is
        # converted to time through the critical timestep setup_problem derived.
        run_duration = duration if duration is not None else n_steps * dem_model.dt
        _apply_time_series_loads(dem_model, loads, run_duration)
        _apply_prescribed_velocities(dem_model, displacements, run_duration)
        dem_model.solve(
            n_steps=n_steps,
            duration=duration,
            update_every=update_every,
            verbose_every=verbose_every,
            convergence_check=convergence_check,
        )
        increments = None
    else:
        block_indices, control, increment = _gradual_actuator(loads, displacements, n_increments)
        # The actuator owns the loads of the blocks it controls; every other
        # block carries its total load as a constant wrench from t=0.
        _apply_constant_loads(dem_model, loads, skip=block_indices if control == "force" else ())
        records = dem_model.solve_gradual(
            block_indices,
            force_increment=increment if control == "force" else None,
            disp_increment=increment if control == "displacement" else None,
            ramp_time=ramp_time,
            n_increments=n_increments,
            max_steps=max_steps,
            update_every=update_every,
            settle_first=settle_first,
            stop_on_failure=stop_on_failure,
            verbose_every=verbose_every,
        )
        increments = [_increment_data(record) for record in records]

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    results = dem_model.extract_results(step=extract_step)
    # extract_results keys the results to the internal copy of the model and has
    # no problem to point at; re-key them to the pair the caller actually holds.
    results.model_id = str(model.guid)
    results.problem_id = str(problem.guid)
    results.metadata["solver"] = "MasonryDEM"
    results.metadata["mode"] = mode
    if increments is not None:
        results.metadata["increments"] = increments
        results.metadata["converged"] = all(record["converged"] for record in increments)
    return results


# =============================================================================
# Mode selection
# =============================================================================


def select_mode(mode: str, loads: dict, displacements: dict) -> str:
    """Return the solve mode to run, resolving ``"auto"`` against the problem.

    A prescribed displacement or a ramped load is applied incrementally
    (``"gradual"``); a problem that only carries instantaneous loads, or no
    loads at all beyond self-weight, is integrated in one run (``"static"``).

    Parameters
    ----------
    mode : str
        ``"auto"``, ``"static"`` or ``"gradual"``.
    loads : dict
        Centroidal loads, as returned by :func:`~compas_dem.analysis.resolve.resolve_centroidal_loads`.
    displacements : dict
        Prescribed movements, as returned by :func:`~compas_dem.analysis.resolve.resolve_centroidal_displacements`.

    Returns
    -------
    str
        ``"static"`` or ``"gradual"``.
    """
    if mode in ("static", "gradual"):
        return mode
    if mode != "auto":
        raise ValueError(f"mode must be 'auto', 'static' or 'gradual', got {mode!r}.")
    if displacements:
        return "gradual"
    if any(_wrench(entry["by_loading_type"]["ramp"]).any() for entry in loads.values()):
        return "gradual"
    return "static"


def _gradual_actuator(loads: dict, displacements: dict, n_increments: int) -> tuple[list[int], str, list[float]]:
    """Return ``(block_indices, control, increment)`` for the incremental solve.

    ``solve_gradual`` drives its blocks as one actuator, so they must share a
    single increment. A prescribed displacement wins over a ramped load: it is
    the stronger control, and the two cannot be imposed at once.
    """
    if n_increments < 1:
        raise ValueError(f"n_increments must be at least 1, got {n_increments}.")

    if displacements:
        entries = {idx: _displacement_vector(entry) for idx, entry in displacements.items()}
        control = "displacement"
        what = "prescribed displacement"
    else:
        entries = {idx: _wrench(entry["by_loading_type"]["ramp"]) for idx, entry in loads.items()}
        entries = {idx: value for idx, value in entries.items() if value.any()}
        control = "force"
        what = "ramped load"

    if not entries:
        raise ValueError("Gradual mode needs a prescribed displacement or a ramped load to drive, but the problem has neither.")

    block_indices = sorted(entries)
    reference = entries[block_indices[0]]
    diverging = [idx for idx in block_indices if not np.allclose(entries[idx], reference)]
    if diverging:
        raise ValueError(
            f"Gradual mode drives every controlled block with one shared increment, but the {what} differs between blocks "
            f"{block_indices[0]} and {diverging[0]}. Give them the same value, or pass mode='static' to apply each load on its own time series."
        )

    return block_indices, control, (reference / n_increments).tolist()


# =============================================================================
# Loads and boundary conditions
# =============================================================================


def _apply_time_series_loads(dem_model, loads: dict, duration: float) -> None:
    """Register the resolved loads as ``[t, x, y, z]`` tables on the DEM model.

    Sampled at ``[0, 0.98T, T]``, the same convention the LMGC90 backend uses: a
    ramped value ``r`` grows 0 -> r and holds, an instantaneous value ``i`` is
    applied at t=0 and released at the end, and a block carrying both gets the
    single series ``[i, r + i, r]`` rather than one loading type winning outright.
    """
    if duration <= 0.0:
        raise ValueError(f"The static run needs a positive duration to write the load series against, got {duration}.")

    times = [0.0, duration * 0.98, duration]

    for idx, entry in loads.items():
        ramped = _wrench(entry["by_loading_type"]["ramp"])
        instantaneous = _wrench(entry["by_loading_type"]["instantaneous"])
        if not ramped.any() and not instantaneous.any():
            continue
        samples = np.array([instantaneous, ramped + instantaneous, ramped])
        if samples[:, :3].any():
            dem_model.apply_force(idx, np.column_stack([times, samples[:, :3]]))
        if samples[:, 3:].any():
            dem_model.apply_moment(idx, np.column_stack([times, samples[:, 3:]]))


def _apply_constant_loads(dem_model, loads: dict, skip=()) -> None:
    """Register the resolved loads as constant wrenches, active from t=0.

    Gradual mode ramps its own actuator, so everything it does not control is a
    dead load for the whole analysis, regardless of its loading type.
    """
    skip = set(skip)
    wrenches = [(idx, _wrench(entry)) for idx, entry in loads.items() if idx not in skip]
    dem_model.apply_forces([(idx, wrench) for idx, wrench in wrenches if wrench.any()])


def _apply_prescribed_velocities(dem_model, displacements: dict, duration: float) -> None:
    """Impose prescribed movements as constant velocities held over ``duration``.

    ``masonry_dem`` prescribes velocities per block, not per component, so an
    unconstrained component (``None``) is imposed as zero rather than left free.
    """
    if not displacements:
        return
    if duration <= 0.0:
        raise ValueError(f"Prescribed displacements need a positive duration to be imposed over, got {duration}.")

    for idx, entry in displacements.items():
        vector = _displacement_vector(entry) / duration
        rows = [[0.0, *vector[:3]], [duration, *vector[:3]]]
        angular = [[0.0, *vector[3:]], [duration, *vector[3:]]]
        dem_model.apply_velocity(idx, lin_series=rows, ang_series=angular if vector[3:].any() else None)


def _warn_about_pinned_displacements(model: BlockModel, displacements: dict) -> None:
    """Report prescribed movements that the model's own support fixity overrides.

    ``masonry_dem`` snaps every support back to its start position at the end of
    each step, so a settlement prescribed on a support block never takes effect.
    """
    supports = {block.graphnode for block in model.elements() if block.is_support}
    pinned = sorted(supports.intersection(displacements))
    if pinned:
        print(f"Warning: blocks {pinned} are supports, so masonry_dem holds them at their start position and their prescribed displacement is ignored.")
        print("Unflag the support (block.is_support = False) to impose the movement instead.")


# =============================================================================
# Helpers
# =============================================================================


def _check_model(model: BlockModel) -> None:
    """Check that the model can be handed to ``masonry_dem`` as it stands."""
    if not any(model.graph.edge_attribute(edge, "contacts") for edge in model.graph.edges()):
        raise ValueError("The model has no contacts. Call model.compute_contacts() before solving.")

    mismatched = [(i, block.graphnode) for i, block in enumerate(model.elements()) if block.graphnode != i]
    if mismatched:
        i, graphnode = mismatched[0]
        raise ValueError(
            f"masonry_dem indexes its blocks by their position in model.elements(), but element {i} sits on graph node {graphnode}. Rebuild the model so the two agree."
        )


def _gravity(problem: Problem) -> float:
    """Return the gravitational acceleration the problem's boundary conditions agree on."""
    values = {group.g for group in problem.boundary_conditions}
    if len(values) > 1:
        raise ValueError(f"The boundary conditions prescribe different gravitational accelerations ({sorted(values)}); masonry_dem applies one to the whole model.")
    return values.pop() if values else 9.81


def _resolve_damping(damping: Union[str, "Damping"], params: Optional[dict]) -> "Damping":
    """Return the damping strategy to install on the model."""
    if isinstance(damping, Damping):
        if params:
            raise ValueError("damping_params applies to a damping model named by string; the given instance is already configured.")
        return damping
    try:
        return DAMPING_MODELS[damping](**(params or {}))
    except KeyError:
        raise ValueError(f"Damping model {damping!r} is not recognised. Available: {sorted(DAMPING_MODELS)}.") from None


def _apply_settings(dem_model, settings: Optional[dict]) -> None:
    """Write the escape-hatch settings onto the DEM model."""
    unknown = sorted(set(settings or ()).difference(_SETTINGS_WHITELIST))
    if unknown:
        raise ValueError(f"Unknown masonry_dem settings {unknown}. Available: {sorted(_SETTINGS_WHITELIST)}.")
    for name, value in (settings or {}).items():
        setattr(dem_model, name, value)


def _wrench(entry: dict) -> np.ndarray:
    """Return a resolved ``{"force", "moment"}`` pair as one ``[F, M]`` array."""
    return np.array([*entry["force"], *entry["moment"]], dtype=float)


def _displacement_vector(entry: dict) -> np.ndarray:
    """Return a resolved displacement as one ``[translation, rotation]`` array."""
    components = (entry.get("translation") or [None] * 3) + (entry.get("rotation") or [None] * 3)
    return np.array([0.0 if value is None else value for value in components], dtype=float)


def _increment_data(record) -> dict:
    """Return one ``IncrementRecord`` as JSON-serializable metadata."""
    return {
        "index": int(record.index),
        "control": record.control,
        "displacement": [float(value) for value in record.displacement],
        "reaction": [float(value) for value in record.reaction],
        "ufr": float(record.ufr),
        "converged": bool(record.converged),
        "steps": int(record.steps),
    }
