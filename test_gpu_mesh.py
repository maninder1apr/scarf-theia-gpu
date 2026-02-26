#!/usr/bin/env python
# ============================================================
# SCARF (.npz) → hp.Mesh → THEIA Scene → GPU OPTICS
# ============================================================

import numpy as np
from pathlib import Path

import hephaistos as hp
import hephaistos.pipeline as pl

import theia
import theia.units as u

# ============================================================
# 0. GPU CHECK
# ============================================================
print("Using GPU:", hp.getCurrentDevice())
if not hp.isRaytracingSupported():
    raise RuntimeError("GPU ray tracing NOT supported")

hp.enableRaytracing()
print("✅ GPU ray tracing enabled")

# ============================================================
# 1. PATHS + LOAD SCARF MESHES
# ============================================================

BASE = Path("/home/users01/msingh/TheiaSimulations/MeshTest/meshes")
PLACEMENTS_FILE = BASE / "placements.npz"

MESH_TABLE = {
    0: BASE / "lar/lar_s.npz",
    1: BASE / "bege/bege_lv.npz",
    2: BASE / "icpc/icpc_lv.npz",
    3: BASE / "pen/pen_bege_pc_s.npz",
    4: BASE / "pen/pen_icpc_pc_s.npz",
}

mesh_cache = {}

for mid, path in MESH_TABLE.items():
    data = np.load(path)

    mesh = hp.Mesh()
    mesh.vertices = data["vertices"].astype(np.float32)
    mesh.indices  = data["indices"].astype(np.int32)

    mesh_cache[mid] = mesh

print(f"Loaded {len(mesh_cache)} hp.Mesh objects")

# ============================================================
# 2. FIBER GEOMETRY
# ============================================================
import trimesh

FIBER_STL = Path("../FiberOptic/fiber_temp_2000x1x1.stl")
CAP_STL   = Path("../FiberOptic/end_cap_5x5.stl")

FIBER_CLEARANCE = 30.0  # mm
FIBER_EDGE   = 1.0    # mm
CAP_OFFSET   = 0.22   # mm from fiber end

def stl_to_hp_mesh(path, scale):
    tm = trimesh.load(path, force="mesh")
    V = np.ascontiguousarray(tm.vertices * scale, dtype=np.float32)
    F = np.ascontiguousarray(tm.faces, dtype=np.int32)

    mesh = hp.Mesh()
    mesh.vertices = V
    mesh.indices  = F
    return mesh

fiber_mesh = stl_to_hp_mesh(FIBER_STL, 100)
cap_mesh   = stl_to_hp_mesh(CAP_STL, 100)

# Rotate fiber so its long axis is Z
R_x_to_z = np.array([
    [0, 0, 1],
    [0, 1, 0],
    [-1, 0, 0],
], dtype=np.float32)

fiber_mesh.vertices = fiber_mesh.vertices @ R_x_to_z.T
cap_mesh.vertices   = cap_mesh.vertices   @ R_x_to_z.T

# ============================================================
# 3. LOAD PLACEMENTS
# ============================================================
p = np.load(PLACEMENTS_FILE)
translations = p["translations"].astype(np.float32)
rotations    = p["rotations"].astype(np.float32)
mesh_ids     = p["mesh_ids"].astype(int)

n_instances = len(mesh_ids)
print(f"Loaded {n_instances} SCARF placements")

# ============================================================
# 4. COMPUTE HPGe + PEN RADIAL ENVELOPE
# ============================================================
r_env = 0.0

for i in range(n_instances):
    mid = int(mesh_ids[i])

    if mid not in (1, 2, 3, 4):
        continue

    V = mesh_cache[mid].vertices
    Rm = rotations[i].reshape(3, 3)
    Tm = translations[i]

    Vt = (V @ Rm.T) + Tm
    r = np.sqrt(Vt[:, 0]**2 + Vt[:, 1]**2)

    r_env = max(r_env, r.max())

print(f"✅ HPGe + PEN envelope radius = {r_env:.2f} mm")

R_fiber = r_env + FIBER_CLEARANCE
print(f"✅ Fiber placement radius R_fiber = {R_fiber:.2f} mm")

# ============================================================
# 5. MATERIALS
# ============================================================
from theia.material import DispersionFreeMedium, Material, MaterialStore

lambda_range = (350.0, 550.0) * u.nm

lar_medium = DispersionFreeMedium(n=1.23).createMedium(
    name="lar",
    wavelengthRange=lambda_range
)

ge_medium = DispersionFreeMedium(n=4.0).createMedium(
    name="ge",
    wavelengthRange=lambda_range
)

pen_medium = DispersionFreeMedium(n=1.65).createMedium(
    name="pen",
    wavelengthRange=lambda_range
)

fiber_medium = DispersionFreeMedium(n=1.59).createMedium(
    name="fiber",
    wavelengthRange=lambda_range
)
materialStore = MaterialStore([
    Material("lar", lar_medium, None),
    Material("ge",  ge_medium,  lar_medium),
    Material("pen", pen_medium, lar_medium),
    Material("fiber", fiber_medium, lar_medium),
    Material("detector", None, None, flags="*DB"),
])

# ============================================================
# 6. BUILD SCENE
# ============================================================
from theia.scene import MeshStore, Scene, Transform

meshStore = MeshStore({
    "lar":   mesh_cache[0],
    "ge":    mesh_cache[1],
    "icpc":  mesh_cache[2],
    "pen":   mesh_cache[3],
    "fiber": fiber_mesh,
    "cap":   cap_mesh,
})

mesh_name = {0: "lar", 1: "ge", 2: "icpc", 3: "pen", 4: "pen"}
mat_name  = {0: "lar", 1: "ge", 2: "ge",   3: "pen", 4: "pen"}

instances = []

for i in range(n_instances):
    mid = mesh_ids[i]

    M = np.zeros((3, 4), dtype=np.float32)
    M[:3, :3] = rotations[i].reshape(3, 3)
    M[:3,  3] = translations[i]

    instances.append(
        meshStore.createInstance(
            mesh_name[mid],
            mat_name[mid],
            Transform(M),
        )
    )

N_FIBERS = 100

z_fiber_min = fiber_mesh.vertices[:, 2].min()
z_fiber_max = fiber_mesh.vertices[:, 2].max()

z_cap_top    = z_fiber_max + CAP_OFFSET
z_cap_bottom = z_fiber_min - CAP_OFFSET

cap_extent_x = np.ptp(cap_mesh.vertices[:, 0])
cap_extent_y = np.ptp(cap_mesh.vertices[:, 1])

cap_scale_x = FIBER_EDGE / cap_extent_x
cap_scale_y = FIBER_EDGE / cap_extent_y

cap_scale = Transform.Scale(cap_scale_x, cap_scale_y, 1.0)

for k in range(N_FIBERS):
    phi = 2.0 * np.pi * k / N_FIBERS
    x = R_fiber * np.cos(phi)
    y = R_fiber * np.sin(phi)

    # -------------------------
    # Fiber body
    # -------------------------
    M_fiber = np.zeros((3, 4), dtype=np.float32)
    M_fiber[:3, :3] = np.eye(3)
    M_fiber[:3,  3] = [x, y, 0.0]

    instances.append(
        meshStore.createInstance(
            "fiber",
            "fiber",
            Transform(M_fiber),
        )
    )

    # -------------------------
    # TOP detector
    # -------------------------
    M_top = np.zeros((3, 4), dtype=np.float32)
    M_top[:3, :3] = np.eye(3)
    M_top[:3,  3] = [x, y, z_cap_top]

    instances.append(
        meshStore.createInstance(
            "cap",
            "detector",
            cap_scale @ Transform(M_top),
            detectorId=2*k,          # even IDs
        )
    )

    # -------------------------
    # BOTTOM detector
    # -------------------------
    M_bot = np.zeros((3, 4), dtype=np.float32)
    M_bot[:3, :3] = np.eye(3)
    M_bot[:3,  3] = [x, y, z_cap_bottom]

    instances.append(
        meshStore.createInstance(
            "cap",
            "detector",
            cap_scale @ Transform(M_bot),
            detectorId=2*k + 1,      # odd IDs
        )
    )

scene = Scene(
    instances,
    materialStore,
    medium=materialStore.media["lar"],
)

print("✅ THEIA Scene created")
# ============================================================
# 7. SOURCE + TRACER
# ============================================================
from theia.random import PhiloxRNG
from theia.light import ConstWavelengthSource, SphericalLightSource
from theia.response import IntegratingHitResponse, UniformValueResponse
from theia.trace import SceneForwardTracer

rng = PhiloxRNG(key=0xC0FFEE)

photons = ConstWavelengthSource(420.0 * u.nm)
source  = SphericalLightSource(timeRange=(0.0, 0.0))

response = IntegratingHitResponse(
    UniformValueResponse(),
    detectorCount=2*N_FIBERS,
)

batch_size = 256 * 1024

tracer = SceneForwardTracer(
    batch_size,
    source,
    photons,
    response,
    rng,
    scene,
    maxPathLength=60000,
    scatterCoefficient=0.0,
    maxTime=float("inf"),
)

rng.autoAdvance = tracer.nRNGSamples
print("✅ THEIA tracer initialized")

# ============================================================
# 8. RUN
# ============================================================
results = []

def process(ctx, batch, _):
    results.append(response.result(ctx).copy())

pipeline  = pl.Pipeline(tracer.collectStages())
scheduler = pl.PipelineScheduler(pipeline, processFn=process)

scheduler.schedule([{"lightSource__position": (0.0, 0.0, 0.0)}])
scheduler.wait()

hits = np.array(results)

print("===================================================")
print("THEIA + SCARF OPTICAL TEST COMPLETE")
print(f"Photons simulated : {batch_size:,}")
print(f"Detected photons  : {hits.sum()}")
print(f"Detection eff.    : {hits.sum() / batch_size:.6e}")
print("===================================================")


# ============================================================
# 9. INTERACTIVE VISUALIZATION (matplotlib3d)
# ============================================================

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa
from matplotlib.patches import Patch
import matplotlib.cm as cm

detector_cmap = cm.get_cmap("hsv", 2 * N_FIBERS)

# ------------------------------------------------------------
# Colors & transparency
# ------------------------------------------------------------
MESH_COLORS = {
    0: "#6fa8dc",
    1: "#cc0000",
    2: "#7a1fa2",
    3: "#f6b26b",
    4: "#f6b26b",
}

MESH_ALPHA = {
    0: 0.12,
    1: 0.80,
    2: 0.80,
    3: 0.30,
    4: 0.30,
}

fig = plt.figure(figsize=(14, 12), dpi=150)
ax = fig.add_subplot(111, projection="3d")

all_points = []

# ------------------------------------------------------------
# SCARF geometry
# ------------------------------------------------------------
for i in range(n_instances):
    mid = mesh_ids[i]

    V = mesh_cache[mid].vertices
    F = mesh_cache[mid].indices

    R = rotations[i].reshape(3, 3)
    T = translations[i]

    Vt = (V @ R.T) + T
    all_points.append(Vt)

    ax.plot_trisurf(
    Vt[:, 0], Vt[:, 1], Vt[:, 2],
    triangles=F,
    color=MESH_COLORS[mid],
    alpha=MESH_ALPHA[mid],
    linewidth=0,
    edgecolor="none",
    antialiased=True,
    shade=True,          # ⭐ this is the key
)


# ------------------------------------------------------------
# Fiber centerlines (FAST & SAFE)
# ------------------------------------------------------------
zmin = fiber_mesh.vertices[:, 2].min()
zmax = fiber_mesh.vertices[:, 2].max()


z = np.linspace(zmin, zmax, 200)

for k in range(N_FIBERS):
    phi = 2.0 * np.pi * k / N_FIBERS
    x = R_fiber * np.cos(phi)
    y = R_fiber * np.sin(phi)

    ax.plot(
        np.full_like(z, x),
        np.full_like(z, y),
        z,
        color="#39ff14",
        linewidth=1.5,
        alpha=0.9,
    )

# ------------------------------------------------------------
# Top detector caps
# ------------------------------------------------------------
for k in range(N_FIBERS):
    phi = 2.0 * np.pi * k / N_FIBERS
    x = R_fiber * np.cos(phi)
    y = R_fiber * np.sin(phi)

    Vt = (cap_mesh.vertices * np.array([
              cap_scale_x,
              cap_scale_y,
              1.0
          ])) + np.array([x, y, z_cap_top])

    ax.plot_trisurf(
        Vt[:, 0], Vt[:, 1], Vt[:, 2],
        triangles=cap_mesh.indices,
        color=detector_cmap(2*k),   # even IDs = top
        alpha=0.95,
        linewidth=0,
        antialiased=True,
        shade=True,
    )

    all_points.append(Vt)

# ------------------------------------------------------------
# Bottom detector caps
# ------------------------------------------------------------
for k in range(N_FIBERS):
    phi = 2.0 * np.pi * k / N_FIBERS
    x = R_fiber * np.cos(phi)
    y = R_fiber * np.sin(phi)

    Vt = (cap_mesh.vertices * np.array([
              cap_scale_x,
              cap_scale_y,
              1.0
          ])) + np.array([x, y, z_cap_bottom])

    ax.plot_trisurf(
        Vt[:, 0], Vt[:, 1], Vt[:, 2],
        triangles=cap_mesh.indices,
        color=detector_cmap(2*k + 1),
        alpha=0.95,
        linewidth=0,
        antialiased=True,
        shade=True,
    )

    all_points.append(Vt)


# ------------------------------------------------------------
# Equal aspect ratio
# ------------------------------------------------------------
all_points = np.vstack(all_points)
mins = all_points.min(axis=0)
maxs = all_points.max(axis=0)

center = 0.5 * (mins + maxs)
extent = 0.5 * np.max(maxs - mins)

ax.set_xlim(center[0] - extent, center[0] + extent)
ax.set_ylim(center[1] - extent, center[1] + extent)
ax.set_zlim(center[2] - extent, center[2] + extent)
ax.set_box_aspect([1, 1, 1])

# ------------------------------------------------------------
# Legend & view
# ------------------------------------------------------------
ax.legend(
    handles=[
        Patch(facecolor="#6fa8dc", label="Liquid Argon"),
        Patch(facecolor="#f6b26b", label="PEN"),
        Patch(facecolor="#cc0000", label="BEGe (HPGe)"),
        Patch(facecolor="#7a1fa2", label="ICPC (HPGe)"),
    ],
    frameon=False,
)
# ------------------------------------------------------------
# View, SAVE, then interact
# ------------------------------------------------------------
ax.set_title("SCARF Geometry + Fiber Centerlines")
ax.set_axis_off()

# ------------------------------------------------------------
# FORCE Z VISIBILITY (DEBUG – DO THIS)
# ------------------------------------------------------------
ax.view_init(elev=0, azim=90)   # look along X, see full Z

# Tighten X/Y, emphasize Z
ax.set_xlim(-R_fiber * 1.5, R_fiber * 1.5)
ax.set_ylim(-R_fiber * 1.5, R_fiber * 1.5)
ax.set_zlim(zmin, zmax)

plt.tight_layout()

# ✅ SAVE FIRST (important!)
plt.savefig("scarf_geometry.png", dpi=300)
plt.savefig("scarf_geometry.pdf")
print("✅ Saved scarf_geometry.png and scarf_geometry.pdf")

print("Fiber Z min:", zmin)
print("Fiber Z max:", zmax)
print("Fiber length:", zmax - zmin)

# ✅ THEN SHOW INTERACTIVELY
plt.show()
# ------------------------------------------------------------
# HARD EXIT (prevents matplotlib teardown segfault on HPC)
# ------------------------------------------------------------
import os
os._exit(0)
