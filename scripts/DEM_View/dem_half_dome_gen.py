import pathlib
from math import pi
from math import sin

import compas
from compas.datastructures import Mesh
from compas.geometry import SphericalSurface
from compas_dem.elements import Block
from compas_dem.models import BlockModel
from compas_dem.viewer import DEMViewer

# =============================================================================
# Geometry
# =============================================================================

RADIUS = 5
THICKNESS = 0.16
SPRING = 0.35  # v parameter of the springing: 0 is the crown, 0.5 the equator
NU = 32  # number of blocks around the full circumference at the springing
NV = 15  # number of courses between the springing and the crown
MIN_WIDTH = 0.5  # merge blocks in pairs when a course gets narrower than this
HALF = True  # if True, build only half of the dome, cut through the vertical axis

DU = [0, 0.5] if HALF else [0, 1.0]
SWEEP = DU[1] - DU[0]

surface = SphericalSurface(radius=RADIUS)

dv = SPRING / NV


def extrados(point):
    """Project a point on the intrados radially outward to the extrados."""
    direction = (point - surface.frame.point).unitized()
    return point + direction * THICKNESS


# The rings shrink towards the crown, so every course that would end up with
# blocks narrower than MIN_WIDTH has half as many blocks as the one below it.
counts = []
for k in range(NV - 1):
    v = SPRING - (k + 0.5) * dv
    n = counts[-1] if counts else NU
    while n > 4 and n % 2 == 0 and 2 * pi * RADIUS * sin(pi * v) / n < MIN_WIDTH:
        n //= 2
    counts.append(n)

blocks = []

# Courses are built from the springing upward. Where two courses have a
# different number of blocks, the bed between them is split at the joints of the
# course below, so the two sides always meet facet to facet.
for k in range(NV - 1):
    v_lo = SPRING - k * dv  # ring on the springing side
    v_hi = v_lo - dv  # ring on the crown side
    n = counts[k]
    p = counts[k - 1] // n if k else 1  # bed facets on the springing side
    step = SWEEP / round(n * SWEEP)

    for j in range(round(n * SWEEP)):
        u = DU[0] + j * step
        lower = [surface.point_at(u + i * step / p, v_lo) for i in range(p + 1)]
        upper = [surface.point_at(u, v_hi), surface.point_at(u + step, v_hi)]
        vertices = lower + [extrados(point) for point in lower] + upper + [extrados(point) for point in upper]

        a = list(range(p + 1))  # intrados, springing side
        b = [i + p + 1 for i in a]  # extrados, springing side
        c = [2 * (p + 1), 2 * (p + 1) + 1]  # intrados, crown side
        d = [c[0] + 2, c[1] + 2]  # extrados, crown side

        faces = [[a[i], a[i + 1], b[i + 1], b[i]] for i in range(p)]  # beds towards the springing
        faces.append([c[1], c[0], d[0], d[1]])  # bed towards the crown
        faces.append([c[1]] + a[::-1] + [c[0]])  # intrados
        faces.append([d[0]] + b + [d[1]])  # extrados
        faces.append([c[0], a[0], b[0], d[0]])  # meridian joint
        faces.append([a[p], c[1], d[1], b[p]])  # meridian joint

        blocks.append(Mesh.from_vertices_and_faces(vertices, faces))

# =============================================================================
# Crown stone
# =============================================================================

# The last course closes into a single crown stone: a small spherical cap that
# follows the intrados and the extrados, so the dome keeps its curvature all the
# way to the top instead of being plugged with a flat lid. Its rim sits on the
# joints of the course below.

nu = round(counts[-1] * SWEEP)
step = SWEEP / nu

rim_intrados = [surface.point_at(DU[0] + j * step, dv) for j in range(nu + 1)]

if not HALF:
    rim_intrados.pop()  # the sweep is periodic: the last sample repeats the first

rim_extrados = [extrados(point) for point in rim_intrados]
apex_intrados = surface.point_at(0, 0)
apex_extrados = extrados(apex_intrados)

n = len(rim_intrados)
inner, outer, apex_in, apex_out = 0, n, 2 * n, 2 * n + 1

faces = []
for j in range(nu):
    k = (j + 1) % n
    faces.append([inner + j, inner + k, outer + k, outer + j])  # bed towards the springing
    faces.append([apex_out, outer + j, outer + k])  # extrados
    faces.append([apex_in, inner + k, inner + j])  # intrados
if HALF:
    faces.append([apex_in, inner, outer, apex_out])  # cut face
    faces.append([inner + n - 1, apex_in, apex_out, outer + n - 1])  # cut face

vertices = rim_intrados + rim_extrados + [apex_intrados, apex_extrados]
blocks.append(Mesh.from_vertices_and_faces(vertices, faces))

# Use individual blocks as bricks (mesh.join() has issues)
bricks = []
for i, block in enumerate(blocks):
    brick: Mesh = block.copy()
    brick.attributes["is_support"] = brick.centroid()[2] < 0.4
    bricks.append(brick)

# =============================================================================
# Model and interactions
# =============================================================================

model = BlockModel()

for brick in bricks:
    element = Block.from_mesh(brick)
    element.is_support = brick.attributes["is_support"]
    model.add_element(element)

# =============================================================================

model.compute_contacts(tolerance=0.001)

# =============================================================================
# Export
# =============================================================================

compas.json_dump(model, pathlib.Path(__file__).parent / "dome.json")

# =============================================================================
# Viz
# =============================================================================

viewer = DEMViewer(model)

viewer.setup()
viewer.show()
