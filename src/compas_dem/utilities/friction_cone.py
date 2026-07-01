import matplotlib.pyplot as plt
import numpy as np

from compas_dem.problem.results import Results

_CMAP = plt.get_cmap("tab10")


def plot_friction_cone(
    results: Results,
    title: str = "Friction Cone",
    show: bool = True,
) -> plt.Figure:
    """Plot the Coulomb friction cone from the contact results.

    Each plotted point represents one contact interface, with forces averaged
    over the contact vertices.  Three panels are shown: fn–ft1, fn–ft2, and a
    3D cone view.

    Parameters
    ----------
    results : :class:`~compas_dem.problem.Results`
        Results object returned by the solver.
    title : str
        Figure suptitle.
    show : bool
        Call ``plt.show()`` when done.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> fig = plot_friction_cone(results)
    """
    mu = results.metadata.get("mu")
    if mu is None:
        raise ValueError(
            "No friction coefficient found in results.metadata['mu']. Ensure the solver stored it before calling this function."
        )

    fn_list, ft1_list, ft2_list = [], [], []
    for edge in results.edges():
        fc = results.contact_data(edge)
        if fc is None or not fc.forces:
            continue
        fn_list.append(
            float(np.mean([f.get("c_np", 0.0) - f.get("c_nn", 0.0) for f in fc.forces]))
        )
        ft1_list.append(float(np.mean([f.get("c_u", 0.0) for f in fc.forces])))
        ft2_list.append(float(np.mean([f.get("c_v", 0.0) for f in fc.forces])))

    if not fn_list:
        raise ValueError(
            "No contact_data found in results. Ensure the solver ran successfully."
        )

    fn_avg = np.array(fn_list)
    ft1_avg = np.array(ft1_list)
    ft2_avg = np.array(ft2_list)

    fn_max = max(fn_avg.max() * 1.3, 1e-6)
    fn_line = np.linspace(0, fn_max, 200)

    fig = plt.figure(figsize=(14, 5))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133, projection="3d")

    # --- 2D panels ---
    for ax, ft_avg, ft_label in [
        (ax1, ft1_avg, "$f_{t1}$"),
        (ax2, ft2_avg, "$f_{t2}$"),
    ]:
        ax.fill_between(
            fn_line, mu * fn_line, -mu * fn_line, alpha=0.15, color="steelblue"
        )
        ax.plot(fn_line, mu * fn_line, color="steelblue", lw=1.2)
        ax.plot(fn_line, -mu * fn_line, color="steelblue", lw=1.2)
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.set_xlabel("$f_n$  [N, compression +]")
        ax.set_ylabel(f"{ft_label}  [N]")
        ax.ticklabel_format(style="sci", axis="both", scilimits=(3, 3))
        ax.set_xlim(0, fn_max)
        half = fn_max * mu
        ax.set_ylim(-half * 1.1, half * 1.1)

        inside = (fn_avg >= 0) & (np.abs(ft_avg) <= mu * fn_avg + 1e-10)
        if inside.any():
            ax.scatter(
                fn_avg[inside],
                ft_avg[inside],
                color=_CMAP(0),
                s=30,
                zorder=5,
                label="Interfaces",
            )
        if (~inside).any():
            ax.scatter(
                fn_avg[~inside],
                ft_avg[~inside],
                color=_CMAP(1),
                s=40,
                marker="x",
                lw=1.5,
                zorder=5,
                label="Outside cone",
            )
        ax.legend(fontsize=7, loc="upper left")

    # --- 3D panel ---
    fn_vals = np.linspace(0, fn_max, 50)
    theta = np.linspace(0, 2 * np.pi, 60)
    FN, THETA = np.meshgrid(fn_vals, theta)
    ax3.plot_surface(
        FN,
        mu * FN * np.cos(THETA),
        mu * FN * np.sin(THETA),
        alpha=0.15,
        color="#b0c4de",
        linewidth=0,
    )

    ft_norm = np.sqrt(ft1_avg**2 + ft2_avg**2)
    inside = (fn_avg >= 0) & (ft_norm <= mu * fn_avg + 1e-10)
    if inside.any():
        ax3.scatter(
            fn_avg[inside],
            ft1_avg[inside],
            ft2_avg[inside],
            color=_CMAP(0),
            s=30,
            depthshade=True,
            label="Interfaces",
        )
    if (~inside).any():
        ax3.scatter(
            fn_avg[~inside],
            ft1_avg[~inside],
            ft2_avg[~inside],
            color=_CMAP(1),
            marker="x",
            s=40,
            depthshade=True,
            label="Outside cone",
        )

    half = fn_max * mu
    ax3.set_xlim(0, fn_max)
    ax3.set_ylim(-half, half)
    ax3.set_zlim(-half, half)
    ax3.set_xlabel("$f_n$  [N]", labelpad=6)
    ax3.set_ylabel("$f_{t1}$  [N]", labelpad=6)
    ax3.set_zlabel("$f_{t2}$  [N]", labelpad=6)
    ax3.ticklabel_format(style="sci", axis="both", scilimits=(3, 3))
    ax3.legend(fontsize=7)

    fig.suptitle(f"{title}  (μ = {mu:.3g})", fontsize=12)
    plt.tight_layout()
    if show:
        plt.show()
    return fig
