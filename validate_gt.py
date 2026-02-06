#!/usr/bin/env python3
"""
Validación de Ground Truth (29.4M particles):
A. PDD (Percent Depth Dose) - suavidad vs profundidad
B. Perfiles transversales - simetría en X, Y
C. SNR en periferia - ruido fuera del haz

Uso: python validate_gt.py <path_to_dose_edep.npy>
     o solo: python validate_gt.py (para analizar todos los archivos locales)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import glob

print("=" * 70)
print("VALIDACIÓN DE GROUND TRUTH (29.4M PARTICLES)")
print("=" * 70)

# Buscar archivos dose_edep.npy
if len(sys.argv) > 1:
    # Si se proporciona path específico
    files_to_analyze = [Path(sys.argv[1])]
else:
    # Buscar en directorios comunes
    candidates = []
    candidates.extend(glob.glob("**/dose_edep.npy", recursive=True))
    candidates.extend(glob.glob("simulations/**/dose_edep.npy", recursive=True))
    candidates.append("dose_edep.npy")  # Local root
    
    files_to_analyze = [Path(f) for f in candidates if Path(f).exists()]

if not files_to_analyze:
    print("❌ No se encontraron archivos dose_edep.npy")
    print("   Uso: python validate_gt.py <path_to_dose_edep.npy>")
    sys.exit(1)

for dose_path in files_to_analyze:
    target_name = str(dose_path.parent.name) if dose_path.parent.name != '.' else 'local_dose'
    
    if not dose_path.exists():
        print(f"\n❌ No existe: {dose_path}")
        continue
    
    print(f"\n📊 Analizando {target_name}...")
    
    # Cargar
    dose = np.load(str(dose_path)).astype(np.float32)
    print(f"   Shape: {dose.shape}, dtype: {dose.dtype}")
    print(f"   Min: {dose.min():.6e}, Max: {dose.max():.6e}")
    
    z, y, x = dose.shape
    
    # ---- A. PDD (Percent Depth Dose) ----
    # Centro del haz aproximado en X, Y
    x_center = x // 2
    y_center = y // 2
    
    pdd = dose[:, y_center, x_center]  # Perfil profundidad en el centro
    pdd_norm = pdd / np.max(pdd) * 100  # Normalizar a 100% en máximo
    
    # Encontrar d_max (profundidad de dosis máxima)
    d_max_idx = np.argmax(pdd)
    d_max = pdd_norm[d_max_idx]
    
    # Calcular suavidad: std de diferencias consecutivas
    pdd_diff = np.diff(pdd_norm)
    pdd_smoothness = np.std(pdd_diff)
    
    print(f"   PDD d_max: índice {d_max_idx}, valor {d_max:.1f}%")
    print(f"   PDD smoothness (std de diferencias): {pdd_smoothness:.4f}")
    print(f"      (< 0.5 = muy suave, 0.5-2 = OK, > 2 = ruidoso)")
    
    # ---- B. Perfiles transversales (simetría) ----
    # Perfil en X en el plano de d_max
    profile_x = dose[d_max_idx, y_center, :]
    profile_x_norm = profile_x / np.max(profile_x) * 100
    
    # Perfil en Y en el plano de d_max
    profile_y = dose[d_max_idx, :, x_center]
    profile_y_norm = profile_y / np.max(profile_y) * 100
    
    # Simetría: comparar mitad izquierda vs derecha
    x_mid = len(profile_x_norm) // 2
    y_mid = len(profile_y_norm) // 2
    
    asymmetry_x = np.mean(np.abs(profile_x_norm[:x_mid] - profile_x_norm[x_mid:][::-1]))
    asymmetry_y = np.mean(np.abs(profile_y_norm[:y_mid] - profile_y_norm[y_mid:][::-1]))
    
    print(f"   Asimetría X: {asymmetry_x:.4f}% (< 5% = simétrico)")
    print(f"   Asimetría Y: {asymmetry_y:.4f}% (< 5% = simétrico)")
    
    # ---- C. SNR en periferia ----
    # Ruido fuera del haz (valores donde dosis es baja)
    # Usar threshold: zona donde dose < 10% del máximo
    threshold = np.max(dose) * 0.1
    
    periphery = dose[dose < threshold]
    if len(periphery) > 0:
        periphery_mean = np.mean(periphery)
        periphery_std = np.std(periphery)
        snr = periphery_mean / (periphery_std + 1e-10)
        
        print(f"   Periferia (< 10% de máximo):")
        print(f"      Media: {periphery_mean:.6e}, Std: {periphery_std:.6e}")
        print(f"      SNR: {snr:.4f} (alto = menos ruido aleatorio)")
    
    # ---- Gráficos ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{target_name} - Validación de GT', fontsize=14, fontweight='bold')
    
    # 1. PDD
    ax = axes[0, 0]
    ax.plot(pdd_norm, 'b-', linewidth=2, label='PDD')
    ax.axvline(d_max_idx, color='r', linestyle='--', alpha=0.7, label=f'd_max (idx={d_max_idx})')
    ax.set_xlabel('Profundidad (vóxeles)')
    ax.set_ylabel('Dosis (%)')
    ax.set_title(f'PDD - Suavidad={pdd_smoothness:.4f}')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 2. Perfil X
    ax = axes[0, 1]
    ax.plot(profile_x_norm, 'g-', linewidth=2, label='Perfil X')
    ax.axvline(x_center, color='r', linestyle='--', alpha=0.7, label='Centro')
    ax.set_xlabel('X (vóxeles)')
    ax.set_ylabel('Dosis (%)')
    ax.set_title(f'Perfil X (Z={d_max_idx}) - Asimetría={asymmetry_x:.4f}%')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 3. Perfil Y
    ax = axes[1, 0]
    ax.plot(profile_y_norm, 'orange', linewidth=2, label='Perfil Y')
    ax.axvline(y_center, color='r', linestyle='--', alpha=0.7, label='Centro')
    ax.set_xlabel('Y (vóxeles)')
    ax.set_ylabel('Dosis (%)')
    ax.set_title(f'Perfil Y (Z={d_max_idx}) - Asimetría={asymmetry_y:.4f}%')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 4. Mapa 2D en d_max
    ax = axes[1, 1]
    slice_2d = dose[d_max_idx, :, :]
    im = ax.imshow(slice_2d, cmap='hot', origin='lower')
    ax.plot(x_center, y_center, 'b+', markersize=15, markeredgewidth=2, label='Centro')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Slice en d_max (Z={d_max_idx})')
    plt.colorbar(im, ax=ax)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"gt_validation_{target_name}.png", dpi=150, bbox_inches='tight')
    print(f"   ✅ Gráfico guardado: gt_validation_{target_name}.png")

print("\n" + "=" * 70)
print("VEREDICTO SOBRE CALIDAD DE GT")
print("=" * 70)
print("✅ BUENO si:")
print("   - PDD smoothness < 1.0 (curva suave)")
print("   - Asimetría X,Y < 5% (perfil simétrico)")
print("   - SNR > 1 (periferia sin ruido excesivo)")
print("\n❌ PROBLEMAS si:")
print("   - PDD muy ruidosa (smoothness > 2)")
print("   - Asimetría > 10% (sesgos sistemáticos)")
print("   - Muchos 'hot pixels' aislados fuera del haz")
