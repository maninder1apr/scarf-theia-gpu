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
# 1. PATHS
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

# ============================================================
# 2. LOAD PLACEMENTS
# ============================================================
p = np.load(PLACEMENTS_FILE)
translations = p["translations"].astype(np.float32)
rotations    = p["rotations"].astype(np.float32)
mesh_ids     = p["mesh_ids"].astype(int)

n_instances = len(mesh_ids)
print(f"Loaded {n_instances} SCARF placements")

# ============================================================
# 3. LOAD SCARF MESHES (hp.Mesh ONLY)
# ============================================================
mesh_cache = {}

for mid, path in MESH_TABLE.items():
    data = np.load(path)

    mesh = hp.Mesh()
    mesh.vertices = data["vertices"].astype(np.float32)
    mesh.indices  = data["indices"].astype(np.int32)

    mesh_cache[mid] = mesh

print(f"Loaded {len(mesh_cache)} hp.Mesh objects")

# ============================================================
# 4. MATERIALS (KNOWN-GOOD API)
# ============================================================
from theia.material import DispersionFreeMedium, Material, MaterialStore

lambda_range = (350.0, 550.0) * u.nm

lar_medium = DispersionFreeMedium(n=1.23).createMedium(
    name="lar", wavelengthRange=lambda_range
)
ge_medium = DispersionFreeMedium(n=4.0).createMedium(
    name="ge", wavelengthRange=lambda_range
)
pen_medium = DispersionFreeMedium(n=1.65).createMedium(
    name="pen", wavelengthRange=lambda_range
)

materialStore = MaterialStore([
    Material("lar", lar_medium, None),
    Material("ge",  ge_medium,  lar_medium),
    Material("pen", pen_medium, lar_medium),
    Material("detector", None, None, flags="*DB"),
])

# ============================================================
# 5. BUILD THEIA SCENE (hp.Mesh → MeshStore)
# ============================================================
from theia.scene import MeshStore, Scene, Transform

meshStore = MeshStore({
    "lar":  mesh_cache[0],
    "ge":   mesh_cache[1],
    "icpc": mesh_cache[2],
    "pen":  mesh_cache[3],  # reuse for both PEN meshes
})

mesh_name = {0: "lar", 1: "ge", 2: "icpc", 3: "pen", 4: "pen"}
mat_name  = {0: "lar", 1: "ge", 2: "ge",   3: "pen", 4: "pen"}

instances = []

for i in range(n_instances):
    mid = mesh_ids[i]
    R = rotations[i].reshape(3, 3)
    T = translations[i]

    M = np.zeros((3, 4), dtype=np.float32)
    M[:3, :3] = R
    M[:3,  3] = T

    instances.append(
        meshStore.createInstance(
            mesh_name[mid],
            mat_name[mid],
            Transform(M),
        )
    )

scene = Scene(
    instances,
    materialStore,
    medium=materialStore.media["lar"],
)

print("✅ THEIA Scene created")

# ============================================================
# 6. SOURCE + TRACER
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
    detectorCount=1,
)

batch_size = 256 * 1024
max_path_length = 60000

tracer = SceneForwardTracer(
    batch_size,
    source,
    photons,
    response,
    rng,
    scene,
    maxPathLength=max_path_length,
    scatterCoefficient=0.0,
    maxTime=float("inf"),
)

rng.autoAdvance = tracer.nRNGSamples
print("✅ THEIA tracer initialized")

# ============================================================
# 7. RUN
# ============================================================
results = []

def process(ctx, batch, _):
    results.append(response.result(ctx).copy())

pipeline  = pl.Pipeline(tracer.collectStages())
scheduler = pl.PipelineScheduler(pipeline, processFn=process)

scheduler.schedule([
    {"lightSource__position": (0.0, 0.0, 0.0)}
])
scheduler.wait()

# ============================================================
# 8. RESULTS
# ============================================================
hits = np.array(results)

print("===================================================")
print("THEIA + SCARF OPTICAL TEST COMPLETE")
print(f"Photons simulated : {batch_size:,}")
print(f"Detected photons  : {hits.sum()}")
print(f"Detection eff.    : {hits.sum() / batch_size:.6e}")
print("===================================================")
# ============================================================
# 9. VISUALIZATION (GEOMETRY + DETECTORS)
# ============================================================
# ============================================================
# 9. VISUALIZATION (SCARF GEOMETRY, HEADLESS, SAVE ONLY)
# ============================================================

import matplotlib
matplotlib.use("Agg")  # REQUIRED on HPC / headless nodes

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
import numpy as np

# ------------------------------------------------------------
# Per-mesh color & transparency (by SCARF mesh_id)
# ------------------------------------------------------------
# mesh_id:
#   0 = LAr
#   1 = BEGe (HPGe)
#   2 = ICPC (HPGe)
#   3 = PEN (BEGe side)
#   4 = PEN (ICPC side)

MESH_COLORS = {
    0: "#6fa8dc",   # Liquid Argon (blue)
    1: "#cc0000",   # BEGe (red)
    2: "#7a1fa2",   # ICPC (purple)
    3: "#f6b26b",   # PEN (orange)
    4: "#f6b26b",   # PEN (orange)
}

MESH_ALPHA = {
    0: 0.12,  # LAr (very transparent)
    1: 0.80,  # BEGe
    2: 0.80,  # ICPC
    3: 0.30,  # PEN
    4: 0.30,  # PEN
}

# ------------------------------------------------------------
# Create figure
# ------------------------------------------------------------
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection="3d")

all_points = []

# ------------------------------------------------------------
# Plot all SCARF instances
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
        Vt[:, 0],
        Vt[:, 1],
        Vt[:, 2],
        triangles=F,
        color=MESH_COLORS.get(mid, "gray"),
        alpha=MESH_ALPHA.get(mid, 0.25),
        linewidth=0,
        edgecolor="none",
    )

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
# Optional: mark optical source at origin
# ------------------------------------------------------------
ax.scatter(
    [0.0], [0.0], [0.0],
    c="black",
    s=90,
    marker="*",
)

# ------------------------------------------------------------
# Legend (manual, reliable)
# ------------------------------------------------------------
from matplotlib.patches import Patch

legend_handles = [
    Patch(facecolor="#6fa8dc", edgecolor="k", label="Liquid Argon"),
    Patch(facecolor="#f6b26b", edgecolor="k", label="PEN"),
    Patch(facecolor="#cc0000", edgecolor="k", label="BEGe (HPGe)"),
    Patch(facecolor="#7a1fa2", edgecolor="k", label="ICPC (HPGe)"),
]

ax.legend(
    handles=legend_handles,
    loc="upper left",
    frameon=False,
)

# ------------------------------------------------------------
# View & formatting
# ------------------------------------------------------------
ax.set_title("SCARF Geometry + Optical Detectors")
ax.set_axis_off()
ax.view_init(elev=20, azim=35)

plt.tight_layout()

# ------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------
plt.savefig("scarf_geometry.pdf")
plt.savefig("scarf_geometry.png", dpi=300)
plt.close(fig)

print("Saved scarf_geometry.pdf and scarf_geometry.png")

# ------------------------------------------------------------
# HARD EXIT (prevents matplotlib teardown segfault on HPC)
# ------------------------------------------------------------
import os
os._exit(0)