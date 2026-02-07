#!/usr/bin/env python3
"""
Analizar distribución de dosis por capa (z-axis)
Ver dónde realmente hay estructura vs donde hay casi 0
"""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

gt = np.load('dose_edep.npy')
D, H, W = gt.shape

print(f"Volumen: {D} × {H} × {W}")
print()
print(f"{'Z':<4} {'Max Dose':<12} {'Mean Dose':<12} {'Nonzero %':<12} {'Status':<30}")
print("-" * 80)

nonzero_by_layer = []

for z in range(D):
    layer = gt[z]
    max_dose = np.max(layer)
    mean_dose = np.mean(layer)
    nonzero_count = np.sum(layer > 0)
    nonzero_pct = 100 * nonzero_count / (H * W)
    
    nonzero_by_layer.append(nonzero_pct)
    
    status = ""
    if max_dose < 1:
        status = "⚠️  MUY BAJA (casi 0)"
    elif max_dose < 10:
        status = "⚠️  BAJA"
    elif max_dose < 100:
        status = "✓ MEDIA"
    else:
        status = "✓✓ ALTA"
    
    print(f"{z:<4d} {max_dose:<12.6f} {mean_dose:<12.6f} {nonzero_pct:<12.2f} {status:<30}")

nonzero_by_layer = np.array(nonzero_by_layer)

print()
print("="*80)
print("ANÁLISIS DE DOSIS SIGNIFICATIVA")
print("="*80)

# Encontrar dónde la dosis es casi 0
thresholds = [1, 5, 10, 50]

for threshold in thresholds:
    # Capas donde max_dose >= threshold
    significant_layers = []
    for z in range(D):
        if np.max(gt[z]) >= threshold:
            significant_layers.append(z)
    
    if significant_layers:
        first_z = significant_layers[0]
        last_z = significant_layers[-1]
        n_layers = len(significant_layers)
        print(f"\nDosis ≥ {threshold}:")
        print(f"  Capas con estructura: {n_layers}/300")
        print(f"  Rango: z={first_z} a z={last_z}")
        print(f"  Volumen: {n_layers} capas")
        print(f"  % del total: {100*n_layers/D:.1f}%")

# Crear figura
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Max dose por capa
axes[0, 0].bar(range(D), [np.max(gt[z]) for z in range(D)], color='steelblue', alpha=0.7)
axes[0, 0].set_xlabel('Z layer')
axes[0, 0].set_ylabel('Max Dose')
axes[0, 0].set_title('Máxima dosis por capa')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Voxeles no-zero por capa
axes[0, 1].bar(range(D), nonzero_by_layer, color='coral', alpha=0.7)
axes[0, 1].set_xlabel('Z layer')
axes[0, 1].set_ylabel('% Nonzero Voxels')
axes[0, 1].set_title('Porcentaje de voxeles con dosis > 0')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: PDD profile
pdd = np.array([np.max(gt[z]) for z in range(D)])
axes[1, 0].plot(pdd, linewidth=2, color='green')
axes[1, 0].fill_between(range(D), 0, pdd, alpha=0.3)
axes[1, 0].set_xlabel('Z layer')
axes[1, 0].set_ylabel('PDD (max dose)')
axes[1, 0].set_title('Percent Depth Dose Profile')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Cumulative nonzero voxels
cumsum = np.cumsum(nonzero_by_layer)
axes[1, 1].plot(cumsum, linewidth=2, color='purple')
axes[1, 1].fill_between(range(D), 0, cumsum, alpha=0.3)
axes[1, 1].set_xlabel('Z layer')
axes[1, 1].set_ylabel('Cumulative % Nonzero')
axes[1, 1].set_title('Acumulativo de voxeles con dosis')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('dose_distribution_analysis.png', dpi=150, bbox_inches='tight')
print("\n✓ Gráfico guardado: dose_distribution_analysis.png")

# Estadísticas agresivas
print()
print("="*80)
print("SUGERENCIA: FILTRADO MÁS AGRESIVO")
print("="*80)

# Encontrar dónde empieza a haber dosis < 1 consistentemente
max_per_z = np.array([np.max(gt[z]) for z in range(D)])
z_with_low_dose = np.where(max_per_z < 1)[0]

if len(z_with_low_dose) > 0:
    print(f"\nCapas con dosis MAX < 1: {len(z_with_low_dose)} capas")
    print(f"  Índices: {sorted(z_with_low_dose)}")

z_with_medium_dose = np.where(max_per_z < 10)[0]
if len(z_with_medium_dose) > 0:
    print(f"\nCapas con dosis MAX < 10: {len(z_with_medium_dose)} capas")
    print(f"  % del volumen: {100*len(z_with_medium_dose)/D:.1f}%")

z_with_significant = np.where(max_per_z >= 10)[0]
if len(z_with_significant) > 0:
    first_sig = z_with_significant[0]
    last_sig = z_with_significant[-1]
    print(f"\nCapas con dosis MAX ≥ 10 (SIGNIFICATIVA):")
    print(f"  Rango: z={first_sig} a z={last_sig}")
    print(f"  Total: {len(z_with_significant)} capas ({100*len(z_with_significant)/D:.1f}%)")
    print(f"  ➜ Podrías filtrar layers z<{first_sig} y z>{last_sig} para ser más agresivo")
