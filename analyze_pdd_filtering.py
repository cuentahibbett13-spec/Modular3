#!/usr/bin/env python3
"""
Calcular cuántos voxeles se eliminan al filtrar 1% inferior de PDD
"""
import numpy as np
from pathlib import Path

def calculate_pdd(vol):
    """Calcula PDD (máximo por layer)"""
    D = vol.shape[0]
    pdd = np.array([np.max(vol[z]) for z in range(D)])
    return pdd

def analyze_pdd_filtering(gt_path, percentile=1.0):
    """Analizar impacto del filtrado de PDD"""
    
    # Cargar GT
    gt = np.load(gt_path)
    D, H, W = gt.shape
    total_voxels = D * H * W
    total_nonzero = np.sum(gt > 0)
    
    print(f"Ground Truth: {D} × {H} × {W} = {total_voxels:,} voxeles totales")
    print(f"Voxeles con dosis > 0: {total_nonzero:,}")
    print()
    
    # Calcular PDD
    pdd = calculate_pdd(gt)
    
    # Umbral para filtrado
    threshold = np.percentile(pdd, percentile)
    
    print(f"PDD Statistics:")
    print(f"  Min: {np.min(pdd):.6f}")
    print(f"  Max: {np.max(pdd):.6f}")
    print(f"  Mean: {np.mean(pdd):.6f}")
    print(f"  Percentil {percentile}%: {threshold:.6f}")
    print()
    
    # Capas a eliminar
    mask_keep = pdd >= threshold
    mask_remove = pdd < threshold
    
    n_keep = np.sum(mask_keep)
    n_remove = np.sum(mask_remove)
    pct_layers_keep = 100 * n_keep / D
    pct_layers_remove = 100 * n_remove / D
    
    print(f"PDD Filtering (percentil {percentile}%):")
    print(f"  Capas a MANTENER: {n_keep}/{D} ({pct_layers_keep:.2f}%)")
    print(f"  Capas a ELIMINAR: {n_remove}/{D} ({pct_layers_remove:.2f}%)")
    print()
    
    # Contar voxeles en capas eliminadas
    voxels_removed = 0
    voxels_nonzero_removed = 0
    
    for z in range(D):
        if not mask_keep[z]:
            voxels_removed += H * W
            voxels_nonzero_removed += np.sum(gt[z] > 0)
    
    pct_voxels_removed = 100 * voxels_removed / total_voxels
    pct_nonzero_removed = 100 * voxels_nonzero_removed / total_nonzero if total_nonzero > 0 else 0
    
    print(f"Voxeles Eliminados:")
    print(f"  Total voxeles: {voxels_removed:,}/{total_voxels:,} ({pct_voxels_removed:.3f}%)")
    print(f"  Voxeles con dosis: {voxels_nonzero_removed:,}/{total_nonzero:,} ({pct_nonzero_removed:.3f}%)")
    print()
    
    # Detalle por capa eliminada
    print(f"Capas Eliminadas (PDD < {threshold:.6f}):")
    print(f"{'Layer':<6} {'PDD':<12} {'Nonzero Voxels':<16}")
    print("-" * 40)
    for z in range(D):
        if not mask_keep[z]:
            nonzero_in_layer = np.sum(gt[z] > 0)
            print(f"{z:<6d} {pdd[z]:<12.6f} {nonzero_in_layer:<16d}")
    
    print()
    print("="*60)
    print("RESUMEN")
    print("="*60)
    print(f"Voxeles totales eliminados: {pct_voxels_removed:.3f}%")
    print(f"Voxeles con dosis eliminados: {pct_nonzero_removed:.3f}%")
    print(f"Impacto en datos: MÍNIMO (< 1% del volumen)")
    print("="*60)

if __name__ == '__main__':
    gt_path = Path('dose_edep.npy')
    
    if not gt_path.exists():
        print(f"❌ No encontrado: {gt_path}")
        exit(1)
    
    analyze_pdd_filtering(str(gt_path), percentile=1.0)
