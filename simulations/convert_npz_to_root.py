#!/usr/bin/env python3
"""
Convertir NPZ IAEA a ROOT para PhaseSpaceSource de OpenGate.
"""

import numpy as np
from pathlib import Path
import uproot

print("="*70)
print("CONVERSIÓN NPZ → ROOT PARA PHASESPACESOURCE")
print("="*70)

# Cargar NPZ
npz_file = 'data/IAEA/Varian_Clinac_2100CD_6MeV_15x15.npz'

print(f"\n📂 Cargando: {npz_file}")
data = np.load(npz_file)

pos_x = data['pos_x']      # cm
pos_y = data['pos_y']      # cm
pos_z = data['pos_z']      # cm
dir_u = data['dir_u']
dir_v = data['dir_v']
dir_w = data['dir_w']
energy = data['energy']    # MeV
weight = data['weight']
pdg = data['pdg']

print(f"✅ Cargados {len(energy):,} partículas")
print(f"   Energía: {energy.mean():.3f} ± {energy.std():.3f} MeV")

# Filtrar electrones E>5.5 MeV y limitar a 500k
mask = (pdg == 11) & (energy > 5.5)
mask_idx = np.where(mask)[0]
if len(mask_idx) > 500000:
    mask_idx = mask_idx[:500000]

print(f"\n🔍 Electrones E>5.5 MeV (usando): {len(mask_idx):,}")

# Guardar como ROOT (TTree) para PhaseSpaceSource
root_file = 'data/IAEA/phsp_500k.root'
print(f"\n💾 Guardando ROOT: {root_file}")

data_dict = {
    'x': pos_x[mask_idx].astype(np.float32),
    'y': pos_y[mask_idx].astype(np.float32),
    'z': pos_z[mask_idx].astype(np.float32),
    'dx': dir_u[mask_idx].astype(np.float32),
    'dy': dir_v[mask_idx].astype(np.float32),
    'dz': dir_w[mask_idx].astype(np.float32),
    'E': energy[mask_idx].astype(np.float32),
    'w': weight[mask_idx].astype(np.float32),
    'pid': (np.ones_like(mask_idx, dtype=np.int32) * 11),
}

with uproot.recreate(root_file) as f:
    f['phsp'] = data_dict

print(f"✅ ROOT guardado")
print(f"\n✅ Listo para PhaseSpaceSource!")
