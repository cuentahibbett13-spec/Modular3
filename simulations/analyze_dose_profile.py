#!/usr/bin/env python3
"""
Analiza perfil de dosis MHD y corrige el cero en la superficie del agua.
"""

import numpy as np
from pathlib import Path
import json

mhd = Path("output_phsp_500k/dose_z_edep.mhd")
raw = Path("output_phsp_500k/dose_z_edep.raw")

if not mhd.exists():
    raise SystemExit("No se encontró dose_z_edep.mhd")

# Parse MHD header
header = {}
for line in mhd.read_text().splitlines():
    if "=" in line:
        k, v = line.split("=", 1)
        header[k.strip()] = v.strip()

sizes = list(map(int, header.get("DimSize").split()))
spacing = list(map(float, header.get("ElementSpacing").split()))
eltype = header.get("ElementType")
raw_file = header.get("ElementDataFile")

# Map ElementType
map_dtype = {
    "MET_FLOAT": np.float32,
    "MET_DOUBLE": np.float64,
    "MET_SHORT": np.int16,
    "MET_USHORT": np.uint16,
    "MET_INT": np.int32,
    "MET_UINT": np.uint32,
}

dtype = map_dtype.get(eltype)
if dtype is None:
    raise SystemExit(f"ElementType no soportado: {eltype}")

raw_path = raw if raw.exists() else mhd.parent / raw_file
arr = np.fromfile(raw_path, dtype=dtype).reshape(sizes[::-1])  # z,y,x

nz, ny, nx = arr.shape
ix, iy = nx // 2, ny // 2
profile = arr[:, iy, ix]

# Z (mm)
z = np.arange(nz) * spacing[2]

max_idx = int(np.argmax(profile))
max_val = float(profile[max_idx])

# Estimar superficie de agua: primer bin por encima del 1% del máximo
threshold = 0.01 * max_val
z0_idx = int(np.argmax(profile >= threshold)) if max_val > 0 else 0
z0 = float(z[z0_idx])

# Métricas relativas al agua
zmax = float(z[max_idx] - z0)

r50 = None
if max_val > 0:
    idx_50 = np.where(profile >= 0.5 * max_val)[0]
    if len(idx_50) > 0:
        r50 = float(z[idx_50[-1]] - z0)

fwhm = None
if max_val > 0:
    idx_fwhm = np.where(profile >= 0.5 * max_val)[0]
    if len(idx_fwhm) > 1:
        fwhm = float((z[idx_fwhm[-1]] - z[idx_fwhm[0]]))

print("Z0 (mm):", z0)
print("Zmax_rel (mm):", zmax)
print("R50_rel (mm):", r50)
print("FWHM (mm):", fwhm)

out = {
    "z0_mm": z0,
    "zmax_rel_mm": zmax,
    "r50_rel_mm": r50,
    "fwhm_mm": fwhm,
    "max_val": max_val,
}

with open("output_phsp_500k/analysis_results.json", "w") as f:
    json.dump(out, f, indent=2)
