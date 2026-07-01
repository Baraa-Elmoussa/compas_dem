import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from compas_dem.interactions import EdgeContact
from compas_dem.interactions import FrictionContact
from compas_dem.problem.results import Results


def _eccentricity(fc) -> float:
    """Compute the eccentricity of the contact resultant relative to the polygon boundary.

    Physically, eccentricity = 0 means the resultant acts through the geometric
    centroid of the contact polygon (ideal, fully uniform compression).
    Eccentricity = 1 means the resultant has reached the polygon edge in the
    direction it is offset — this is the hinging onset condition, where one side
    of the interface is about to open.

    Method
    ------
    A ray is cast from the polygon centroid ``cp`` through the force-weighted
    resultant point ``rp`` (the point where the net normal force effectively acts,
    computed by :attr:`FrictionContact.resultantpoint`).  The ray is then
    intersected with each edge of the polygon to find the boundary distance in
    that specific direction.

    For each polygon edge (a → b), the intersection is found by solving:

        cp + t · d̂ = a + s · (b − a)

    where ``d̂`` is the unit direction from ``cp`` to ``rp``, ``t`` is the
    distance along the ray, and ``s ∈ [0, 1]`` is the position along the edge.
    This gives 3 equations (one per spatial axis) with 2 unknowns (t, s).
    The system is overdetermined but consistent because the ray and edge are
    coplanar; ``np.linalg.lstsq`` finds the exact solution robustly for any
    polygon orientation.

    The smallest positive ``t`` with a valid ``s`` is the boundary distance.

    Returns
    -------
    float
        ``dist(cp, rp) / t_boundary``, clamped to 1.0 if the resultant is
        already outside the polygon.
    """
    polygon = fc.polygon
    if polygon is None or len(polygon.points) < 3 or not fc.forces:
        return 0.0

    cp = np.array(polygon.centroid)  # geometric centroid of the contact polygon
    rp = np.array(fc.resultantpoint)  # force-weighted point where the resultant acts
    d = rp - cp
    dist_cr = np.linalg.norm(d)

    if dist_cr < 1e-14:
        return 0.0  # resultant is at the centroid — perfectly symmetric contact

    d_unit = d / dist_cr
    pts = [np.array(p) for p in polygon.points]
    n = len(pts)

    t_boundary = np.inf
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        # Solve: cp + t*d_unit = a + s*(b-a)
        # Rearranged: [d_unit | -(b-a)] * [t, s]^T = a - cp
        # 3 equations, 2 unknowns — consistent because ray and edge are coplanar.
        A = np.column_stack([d_unit, -(b - a)])
        ts, _, _, _ = np.linalg.lstsq(A, a - cp, rcond=None)
        t, s = ts
        # t > 0: intersection is in the forward direction of the ray
        # 0 <= s <= 1: intersection falls on this edge (not its extension)
        if t > 1e-10 and -1e-6 <= s <= 1 + 1e-6:
            t_boundary = min(t_boundary, t)

    if not np.isfinite(t_boundary):
        return 1.0  # resultant already outside the polygon

    return dist_cr / t_boundary


def plot_safety_diagram(
    results: Results,
    show: bool = True,
) -> plt.Figure:
    """Plot the safety space: sliding index vs. eccentricity per contact interface.

    Sliding index: |ft| / (μ·fn) — 1.0 = Coulomb limit.
    Eccentricity:  dist(centroid, resultant) / dist(centroid, polygon edge)
                   along the resultant direction — 1.0 = hinging onset.

    Parameters
    ----------
    results : :class:`~compas_dem.problem.Results`
        Results object returned by the solver.
    show : bool
        Call ``plt.show()`` when done.

    Returns
    -------
    matplotlib.figure.Figure
    """
    mu = results.metadata.get("mu")
    if mu is None:
        raise ValueError(
            "No friction coefficient found in results.metadata['mu']. Ensure the solver stored it before calling this function."
        )

    ecc_vals, sli_vals, labels = [], [], []

    for edge in results.edges():
        fc = results.contact_data(edge)
        if fc is None or not fc.forces:
            continue

        fn = sum(f.get("c_np", 0.0) - f.get("c_nn", 0.0) for f in fc.forces)
        ft = np.sqrt(
            sum(f.get("c_u", 0.0) for f in fc.forces) ** 2
            + sum(f.get("c_v", 0.0) for f in fc.forces) ** 2
        )

        if isinstance(fc, FrictionContact):
            if fc.resultantpoint is None:
                continue
            ecc = _eccentricity(fc)
        elif isinstance(fc, EdgeContact):
            ecc = 1.0
        else:
            continue

        ecc_vals.append(ecc)
        sli_vals.append(ft / (mu * fn) if fn > 1e-10 else np.nan)
        labels.append(f"{edge[0]}-{edge[1]}")

    if not labels:
        raise ValueError(
            "No contact_data found in results. Ensure the solver ran successfully."
        )

    ecc_vals = np.clip(np.array(ecc_vals), 0.0, 1.0)
    sli_vals = np.clip(np.nan_to_num(sli_vals), 0.0, 1.0)
    crit_vals = np.maximum(ecc_vals, sli_vals)

    _cmap = mcolors.LinearSegmentedColormap.from_list(
        "safety", ["#2ecc71", "#f1c40f", "#e74c3c"]
    )

    fig, ax = plt.subplots(figsize=(7, 7))

    # Background: red when either metric → 1
    g = np.linspace(0, 1, 200)
    E, S = np.meshgrid(g, g)
    ax.imshow(
        np.maximum(E, S),
        origin="lower",
        extent=[0, 1, 0, 1],
        aspect="auto",
        cmap=_cmap,
        alpha=0.2,
        vmin=0,
        vmax=1,
    )

    ax.axvline(1.0, color="#e74c3c", lw=1.2, ls="--", alpha=0.8)
    ax.axhline(1.0, color="#e74c3c", lw=1.2, ls="--", alpha=0.8)

    sc = ax.scatter(
        ecc_vals,
        sli_vals,
        s=80,
        c=crit_vals,
        cmap=_cmap,
        vmin=0,
        vmax=1,
        edgecolors=["#c0392b" if c > 0.8 else "#2c3e50" for c in crit_vals],
        linewidths=1.2,
        zorder=5,
        alpha=0.85,
    )

    _ql = dict(fontsize=9, alpha=0.45, ha="center", va="center", style="italic")
    ax.text(0.25, 0.25, "Safe", color="#27ae60", **_ql)
    ax.text(0.75, 0.25, "Hinging risk", color="#e74c3c", **_ql)
    ax.text(0.25, 0.75, "Sliding risk", color="#e74c3c", **_ql)
    ax.text(0.75, 0.75, "Critical", color="#c0392b", fontweight="bold", **_ql)

    fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02).set_label(
        "Criticality  (0 = safe, 1 = failure)", fontsize=8
    )
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Eccentricity  (1 = resultant at polygon edge)", fontsize=9)
    ax.set_ylabel(f"Sliding index  (1 = Coulomb limit, μ = {mu:.3g})", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # Hover tooltip
    annot = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(10, 10),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2c3e50", lw=0.8),
        fontsize=8,
        color="#2c3e50",
    )
    annot.set_visible(False)
    _labels = labels

    def _hover(event):
        if event.inaxes != ax:
            return
        cont, ind = sc.contains(event)
        if cont:
            idx = ind["ind"][0]
            annot.xy = (ecc_vals[idx], sli_vals[idx])
            annot.set_text(
                f"{_labels[idx]}\necc={ecc_vals[idx]:.3f}  sli={sli_vals[idx]:.3f}"
            )
            annot.set_visible(True)
        else:
            annot.set_visible(False)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", _hover)

    plt.tight_layout()
    if show:
        plt.show()
    return fig
