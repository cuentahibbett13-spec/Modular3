#!/usr/bin/env python3
"""
Validación de todos los Ground Truths (29.4M) en el cluster.
Ejecutar en: /lustre/home/acastaneda/Fernando/Modular3/

Genera:
- Análisis cuantitativo (PDD smoothness, simetría, SNR)
- Gráficos comparativos entre todos los targets
- Reporte final
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import glob

print("=" * 80)
print("VALIDACIÓN DE TODOS LOS GROUND TRUTHS (29.4M PARTICLES)")
print("=" * 80)

# Buscar todos los dose_edep.npy en estructura de simulaciones
simulation_dirs = sorted(glob.glob("simulations/pair_*/dose_edep.npy"))

if not simulation_dirs:
    # Alternativa: buscar en estructura de dataset
    simulation_dirs = sorted(glob.glob("dataset_pilot/*/dose_edep.npy"))

if not simulation_dirs:
    print("❌ No se encontraron simulaciones en:")
    print("   - simulations/pair_*/dose_edep.npy")
    print("   - dataset_pilot/*/dose_edep.npy")
    print("\nIntenta: find . -name dose_edep.npy -type f")
    import sys; sys.exit(1)

print(f"\n📁 Encontradas {len(simulation_dirs)} simulaciones\n")

# Almacenar resultados
results = {
    'pair': [],
    'shape': [],
    'max': [],
    'pdd_smoothness': [],
    'asym_x': [],
    'asym_y': [],
    'snr_periph': [],
}

# Figura para gráficos comparativos
fig_comp, axes_comp = plt.subplots(2, 3, figsize=(16, 10))
fig_comp.suptitle('Comparación de Ground Truths (29.4M Particles)', fontsize=16, fontweight='bold')

for idx, sim_path in enumerate(simulation_dirs):
    pair_name = Path(sim_path).parent.name
    
    print(f"[{idx+1}/{len(simulation_dirs)}] {pair_name}...", end=" ", flush=True)
    
    # Cargar
    dose = np.load(str(sim_path)).astype(np.float32)
    z, y, x = dose.shape
    
    # ---- A. PDD ----
    x_center, y_center = x // 2, y // 2
    pdd = dose[:, y_center, x_center]
    pdd_norm = pdd / np.max(pdd) * 100
    pdd_diff = np.diff(pdd_norm)
    pdd_smoothness = np.std(pdd_diff)
    d_max_idx = np.argmax(pdd)
    
    # ---- B. Simetría ----
    profile_x = dose[d_max_idx, y_center, :] / np.max(dose[d_max_idx, y_center, :]) * 100
    profile_y = dose[d_max_idx, :, x_center] / np.max(dose[d_max_idx, :, x_center]) * 100
    
    x_mid = len(profile_x) // 2
    y_mid = len(profile_y) // 2
    asym_x = np.mean(np.abs(profile_x[:x_mid] - profile_x[x_mid:][::-1]))
    asym_y = np.mean(np.abs(profile_y[:y_mid] - profile_y[y_mid:][::-1]))
    
    # ---- C. SNR Periferia ----
    threshold = np.max(dose) * 0.1
    periphery = dose[dose < threshold]
    if len(periphery) > 0:
        snr = np.mean(periphery) / (np.std(periphery) + 1e-10)
    else:
        snr = np.nan
    
    # Guardar
    results['pair'].append(pair_name)
    results['shape'].append(f"{z}x{y}x{x}")
    results['max'].append(dose.max())
    results['pdd_smoothness'].append(pdd_smoothness)
    results['asym_x'].append(asym_x)
    results['asym_y'].append(asym_y)
    results['snr_periph'].append(snr)
    
    # Estado
    pdd_ok = "✅" if pdd_smoothness < 1.0 else "⚠️" if pdd_smoothness < 2.0 else "❌"
    asym_ok = "✅" if asym_x < 5 and asym_y < 5 else "⚠️"
    snr_ok = "✅" if snr > 1 else "⚠️"
    
    print(f"Smoothness={pdd_smoothness:.2f}{pdd_ok} Asym={asym_x:.1f}/{asym_y:.1f}%{asym_ok} SNR={snr:.2f}{snr_ok}")
    
    # Agregar a figura comparativa
    row, col = idx // 3, idx % 3
    ax = axes_comp[row, col]
    ax.plot(pdd_norm, 'b-', linewidth=2)
    ax.axvline(d_max_idx, color='r', linestyle='--', alpha=0.5)
    ax.set_title(f'{pair_name}\nSmooth={pdd_smoothness:.2f}', fontsize=10)
    ax.set_ylabel('Dosis (%)')
    ax.grid(True, alpha=0.3)

# Guardar figura comparativa
plt.tight_layout()
plt.savefig('gt_validation_all_pdds.png', dpi=150, bbox_inches='tight')
print(f"\n✅ Gráfico PDD comparativo: gt_validation_all_pdds.png")

# ---- Tabla resumen ----
print("\n" + "=" * 80)
print("TABLA RESUMEN")
print("=" * 80)

print(f"{'Pair':<20} {'PDD Smooth':<12} {'Asim X%':<10} {'Asim Y%':<10} {'SNR':<8} {'Estado':<20}")
print("-" * 80)

states = []
for i, pair in enumerate(results['pair']):
    pdd_s = results['pdd_smoothness'][i]
    asym_x = results['asym_x'][i]
    asym_y = results['asym_y'][i]
    snr = results['snr_periph'][i]
    
    # Clasificar
    pdd_state = "BUENO" if pdd_s < 1.0 else "OK" if pdd_s < 2.0 else "RUIDOSO"
    asym_state = "BUENO" if asym_x < 5 and asym_y < 5 else "ASIMETRÍA"
    snr_state = "LIMPIO" if snr > 1 else "RUIDOSO"
    
    overall = "✅ EXCELENTE" if (pdd_s < 1.0 and asym_x < 5 and asym_y < 5 and snr > 1) \
         else "✅ BUENO" if (pdd_s < 2.0 and asym_x < 10 and asym_y < 10) \
         else "⚠️ MARGINAL"
    
    states.append(overall)
    
    print(f"{pair:<20} {pdd_s:<12.4f} {asym_x:<10.2f} {asym_y:<10.2f} {snr:<8.3f} {overall:<20}")

# Estadísticas
print("\n" + "=" * 80)
print("ESTADÍSTICAS GLOBALES")
print("=" * 80)

pdd_arr = np.array(results['pdd_smoothness'])
asym_x_arr = np.array(results['asym_x'])
asym_y_arr = np.array(results['asym_y'])
snr_arr = np.array([s for s in results['snr_periph'] if not np.isnan(s)])

print(f"\nPDD Smoothness:")
print(f"  Media: {np.mean(pdd_arr):.4f} ± {np.std(pdd_arr):.4f}")
print(f"  Rango: [{np.min(pdd_arr):.4f}, {np.max(pdd_arr):.4f}]")
print(f"  Estado: {'✅ EXCELENTE (< 1.0)' if np.mean(pdd_arr) < 1.0 else '✅ BUENO (< 2.0)' if np.mean(pdd_arr) < 2.0 else '⚠️ RUIDOSO'}")

print(f"\nAsimetría X:")
print(f"  Media: {np.mean(asym_x_arr):.4f}% ± {np.std(asym_x_arr):.4f}%")
print(f"  Rango: [{np.min(asym_x_arr):.4f}%, {np.max(asym_x_arr):.4f}%]")
print(f"  Estado: {'✅ SIMÉTRICO' if np.mean(asym_x_arr) < 5 else '⚠️ LIGERA ASIMETRÍA' if np.mean(asym_x_arr) < 10 else '❌ ASIMETRÍA SIGNIFICATIVA'}")

print(f"\nAsimetría Y:")
print(f"  Media: {np.mean(asym_y_arr):.4f}% ± {np.std(asym_y_arr):.4f}%")
print(f"  Rango: [{np.min(asym_y_arr):.4f}%, {np.max(asym_y_arr):.4f}%]")
print(f"  Estado: {'✅ SIMÉTRICO' if np.mean(asym_y_arr) < 5 else '⚠️ LIGERA ASIMETRÍA' if np.mean(asym_y_arr) < 10 else '❌ ASIMETRÍA SIGNIFICATIVA'}")

print(f"\nSNR Periferia:")
print(f"  Media: {np.mean(snr_arr):.4f} ± {np.std(snr_arr):.4f}")
print(f"  Rango: [{np.min(snr_arr):.4f}, {np.max(snr_arr):.4f}]")
print(f"  Estado: {'✅ LIMPIO' if np.mean(snr_arr) > 1 else '⚠️ MODERADO' if np.mean(snr_arr) > 0.2 else '❌ MUY RUIDOSO'}")

# Veredicto final
print("\n" + "=" * 80)
print("VEREDICTO FINAL")
print("=" * 80)

excellent_count = states.count("✅ EXCELENTE")
good_count = states.count("✅ BUENO")
marginal_count = states.count("⚠️ MARGINAL")

total = excellent_count + good_count + marginal_count

print(f"\n✅ EXCELENTE: {excellent_count}/{total} ({100*excellent_count/total:.1f}%)")
print(f"✅ BUENO:     {good_count}/{total} ({100*good_count/total:.1f}%)")
print(f"⚠️ MARGINAL:  {marginal_count}/{total} ({100*marginal_count/total:.1f}%)")

if excellent_count + good_count >= total * 0.8:
    print("\n🎯 RECOMENDACIÓN: 29.4M ES SUFICIENTE COMO GROUND TRUTH")
    print("   Usa estos datos para training. Considera post-procesamiento suave")
    print("   en periferia para eliminar hot pixels aislados si SNR < 0.5")
else:
    print("\n⚠️ RECOMENDACIÓN: Algunos targets tienen calidad marginal")
    print("   Considera aumentar a 40-50M partículas para mejor estadística")

print("\n" + "=" * 80)
