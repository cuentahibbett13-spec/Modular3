#!/usr/bin/env python3
"""
Simulación con PhaseSpaceSource - ROOT
"""

import opengate as gate
import numpy as np
from pathlib import Path
import os
import json

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

seed = 123
output_folder = Path('./output_phsp_500k')
output_folder.mkdir(exist_ok=True)

print("="*70)
print("SIMULACIÓN CON PHASESPACES SOURCE - 500k ELECTRONES")
print("="*70)

# Crear simulación
sim = gate.Simulation()
sim.random_seed = seed
sim.number_of_threads = 1
sim.output_dir = str(output_folder)

# Física
sim.physics_manager.physics_list_name = 'QGSP_BIC_EMZ'

# Mundo
world = sim.world
world.size = [100, 100, 150]
world.material = 'G4_AIR'

# Agua
water_log = sim.add_volume('Box', 'water')
water_log.size = [100, 100, 30]
water_log.translation = [0, 0, 15]
water_log.material = 'G4_WATER'
water_log.color = [0, 0.6, 0.8, 0.5]

print(f"\n🔧 Geometría:")
print(f"   Mundo: 100x100x150 cm")
print(f"   Agua: 100x100x30 cm a Z=15cm")

# Source: PhaseSpaceSource desde ROOT
source = sim.add_source('PhaseSpaceSource', 'root_phsp')
source.phsp_file = 'data/IAEA/phsp_500k.root'
source.particle = 'e-'
source.n = 500000
source.position_key_x = 'x'
source.position_key_y = 'y'
source.position_key_z = 'z'
source.direction_key_x = 'dx'
source.direction_key_y = 'dy'
source.direction_key_z = 'dz'
source.energy_key = 'E'
source.weight_key = 'w'
source.PDGCode_key = 'pid'

print(f"\n📊 Source (PhaseSpaceSource):")
print(f"   File: {source.phsp_file}")
print(f"   Particles: {source.n:,}")

# Dosis en profundidad
dose_z = sim.add_actor('DoseActor', 'dose_z')
dose_z.attached_to = 'water'
dose_z.size = [1, 1, 300]
dose_z.spacing = [100, 100, 1]
dose_z.output_filename = 'dose_z.mhd'
dose_z.write_to_disk = True

# Estadísticas
stats_actor = sim.add_actor('SimulationStatisticsActor', 'stats')

print(f"\n🚀 Ejecutando simulación...")
try:
    sim.run()
    print(f"✅ Simulación completada")
except Exception as e:
    print(f"⚠️  Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Procesar
try:
    dose_actor = sim.get_actor('dose_z')
    dose = dose_actor.dose
    
    ix, iy = dose.shape[0]//2, dose.shape[1]//2
    dose_profile = dose[ix, iy, :]
    
    dose_max = np.max(dose_profile)
    zmax_idx = np.argmax(dose_profile)
    zmax_mm = zmax_idx * 1
    
    idx_50 = np.where(dose_profile >= 0.5 * dose_max)[0]
    r50_mm = idx_50[-1] if len(idx_50) > 0 else None
    
    idx_fwhm = np.where(dose_profile >= 0.5 * dose_max)[0]
    fwhm_mm = idx_fwhm[-1] - idx_fwhm[0] if len(idx_fwhm) > 1 else None
    
    print(f"\n" + "="*70)
    print(f"✅ RESULTADOS TG-51")
    print(f"="*70)
    print(f"  Zmax: {zmax_mm:.1f} mm  (ref: ~13-14 mm para 6 MeV)")
    if r50_mm:
        print(f"  R50:  {r50_mm:.1f} mm  (ref: ~26 mm para 6 MeV)")
    if fwhm_mm:
        print(f"  FWHM: {fwhm_mm:.1f} mm  (ref: ~100 mm)")
    print(f"="*70)
    
    stats = sim.get_actor('stats')
    print(f"\n📊 Simulación:")
    print(f"   Eventos: {stats.counts.event_count:,}")
    
    results = {'zmax_mm': float(zmax_mm), 'r50_mm': float(r50_mm), 'fwhm_mm': float(fwhm_mm)}
    with open(output_folder / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Resultados guardados!")
    
except Exception as e:
    print(f"❌ Error procesando: {e}")
    import traceback
    traceback.print_exc()
