# Solver benchmarks: MasonryDEM vs. 3DEC

Six gravity-only cases, each solved with both `Solver.MasonryDEM()`
(in-process) and `Solver.ThreeDEC()` (Itasca 3DEC, external executable),
timed and checked for stability through the same `compas_dem.problem.Problem`
API.

## Cases

| Script                  | Structure                                        | Observed outcome (both solvers agree) |
|-------------------------|---------------------------------------------------|-----------------------------------------|
| `case_stack.py`         | 6 stacked boxes, no eccentricity                   | stands (baseline)                       |
| `case_arch_stands.py`   | semicircular arch, t/R = 0.20                      | stands                                  |
| `case_arch_fails.py`    | same arch, t/R = 0.06 (< Heyman's ~0.108)          | collapses                               |
| `case_barrel_vault.py`  | 38-voussoir barrel vault, t/R = 0.17               | stands                                  |
| `case_cross_vault.py`   | 184-block cross vault (`data/crossvault.obj`)      | **collapses** (see note below)          |
| `case_armadillo.py`     | 399-block Armadillo decomposition (`data/armadillo.obj`) | **collapses** -- solvers disagree on how much (see note below) |

The two arch cases share every parameter except thickness, isolating a real
geometric (hinging) instability rather than a friction or material cheat.

**Cross vault note:** this was expected to stand, like the barrel vault. It
doesn't -- and both solvers agree it doesn't (MasonryDEM settles into a new,
sagging equilibrium; 3DEC's own equilibrium check reports "Equilibrium NOT
reached" with a much larger runaway displacement). Inspecting the geometry
shows why: the digitized stereotomy only has 4 blocks at the lowest springing
level (a Gothic quadripartite vault carrying thrust to 4 corner piers via
diagonal ribs, not a perimeter-wall-supported shell), and a friction sweep
(mu 0.6 -> 0.9) didn't change the outcome, so this is a geometric/thickness
instability rather than a sliding failure or a benchmark-harness bug. It's
left as-is rather than patched into standing, since both engines independently
reaching the same verdict on a non-trivial real structure is itself a useful
cross-solver agreement check.

**Armadillo note:** this is dry-stacked, uncemented, arbitrarily-shaped
blocks with plain Coulomb friction and no compression-only load path by
design (it's a stress test for contact-graph size/irregularity, not a
structural typology), so both solvers reporting collapse is expected. What's
notable is they disagree sharply on *how much*: over the same 0.3 s
simulated window MasonryDEM's free fall runs away roughly 10x further than
3DEC's (~0.9 m vs. ~0.08 m max displacement), and MasonryDEM is markedly
slower here too (~5x the wall-clock of 3DEC, whereas it is 3-15x *faster* on
every other case in this suite). This is exactly the kind of divergence a cross-solver benchmark
is meant to surface: on regular masonry (arches, vaults, stacks) the two
engines agree closely and MasonryDEM's in-process, no-executable model wins
on speed; on a large, highly irregular, actively-collapsing contact graph,
the two contact/timestep strategies clearly part ways. Investigating that
gap (contact-update frequency, timestep adaptivity during collapse) is
future work, not something this benchmark script resolves.

## Setup

Use the `masonry` conda environment, which has `compas_dem`, `masonry_dem`
and `compas_3dec` installed editable against these sibling repos:

```
conda activate masonry
```

MasonryDEM runs in-process and needs nothing further. 3DEC is always
attempted through `Solver.ThreeDEC()`, exactly like the other compas_dem
3DEC examples -- `compas_3dec` resolves the executable itself (env vars
below are optional overrides, not requirements):

```powershell
$env:COMPAS_3DEC_EXECUTABLE = "C:/path/to/3dec.exe"   # optional, only if auto-discovery doesn't find it
$env:COMPAS_3DEC_VERSION = "7.0"                       # optional, defaults to "7.0"
```

If `compas_3dec` isn't installed at all, 3DEC is reported as `skipped`. If it
is installed but the solve itself fails (no license, no executable found,
...), that failure is reported as an `error` row instead of crashing the
benchmark.

## Running

```
python case_arch_stands.py      # single case
python run_all.py                # all six, plus a combined summary table
```

`case_cross_vault.py` and `case_armadillo.py` are the two heaviest cases
(184 and 399 blocks). `case_armadillo.py` in particular can take several
minutes for the MasonryDEM row, since it never converges and its irregular,
actively-collapsing contact graph is expensive per step.

## Reading the output

Each case prints one row per backend:

```
Solver      Status      Time [s]   Max disp [m]     Verdict  Note
MasonryDEM  ok             0.842         0.0021        stood
3DEC        skipped           -              -             -  COMPAS_3DEC_EXECUTABLE not set
```

- **Time [s]** is wall-clock time inside `problem.solve()` only.
- **Max disp [m]** is the largest rigid-body displacement magnitude among the
  non-support blocks, from the same `Results.displacement()` accessor for
  both backends.
- **Verdict** is `stood` if that max displacement stays under the case's
  threshold (a few cm — normal elastic contact settling), `collapsed`
  otherwise.
- `MasonryDEM` uses `convergence_check=True`, so a genuinely stable structure
  stops early once forces balance (fast); a collapsing one never converges
  and runs the full simulated `duration` (slow) -- the timing difference
  between `case_arch_stands.py` and `case_arch_fails.py` is itself part of
  the benchmark.

`common.py` holds the shared material/contact/joint setup and the
timing/reporting harness used by every case.
