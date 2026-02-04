#!/usr/bin/env python3
"""
Análisis parametrizado de perfiles de dosis (compatible con cluster).

Uso:
    python analyze_dose_parametrized.py \
        --input output_job_001/dose_z_edep.mhd \
        --output analysis_job_001
"""

import argparse
import numpy as np
from pathlib import Path
import json
import sys
import os

# Backend sin display
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_mhd(mhd_path):
    """Parse MHD header."""
    header = {}
    for line in mhd_path.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            header[k.strip()] = v.strip()
    return header


def analyze_dose_profile(args):
    """Analiza perfil de dosis y genera métricas."""
    
    mhd_path = Path(args.input)
    if not mhd_path.exists():
        print(f"❌ ERROR: No se encuentra {args.input}")
        sys.exit(1)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("ANÁLISIS DE PERFIL DE DOSIS")
    print("="*70)
    print(f"  Input:  {mhd_path}")
    print(f"  Output: {output_dir}")
    print("="*70)
    
    # Parse header
    header = parse_mhd(mhd_path)
    sizes = list(map(int, header.get("DimSize").split()))
    spacing = list(map(float, header.get("ElementSpacing").split()))
    eltype = header.get("ElementType")
    raw_file = header.get("ElementDataFile")
    
    # Map ElementType
    dtype_map = {
        "MET_FLOAT": np.float32,
        "MET_DOUBLE": np.float64,
        "MET_SHORT": np.int16,
        "MET_USHORT": np.uint16,
        "MET_INT": np.int32,
        "MET_UINT": np.uint32,
    }
    
    dtype = dtype_map.get(eltype)
    if dtype is None:
        print(f"❌ ERROR: ElementType no soportado: {eltype}")
        sys.exit(1)
    
    # Leer raw
    raw_path = mhd_path.parent / raw_file
    if not raw_path.exists():
        print(f"❌ ERROR: No se encuentra {raw_path}")
        sys.exit(1)
    
    arr = np.fromfile(raw_path, dtype=dtype).reshape(sizes[::-1])  # z,y,x
    
    nz, ny, nx = arr.shape
    ix, iy = nx // 2, ny // 2
    profile = arr[:, iy, ix]
    
    # Coordenadas Z (mm)
    z = np.arange(nz) * spacing[2]
    
    print(f"\n📊 Datos:")
    print(f"   Shape:   {arr.shape} (Z, Y, X)")
    print(f"   Spacing: {spacing} mm")
    print(f"   Profile: {len(profile)} bins")
    
    # Encontrar máximo
    max_idx = int(np.argmax(profile))
    max_val = float(profile[max_idx])
    
    print(f"\n   Dosis max: {max_val:.3e} @ bin {max_idx}")
    
    # Detectar superficie del agua (1% threshold)
    threshold = 0.01 * max_val
    z0_idx = int(np.argmax(profile >= threshold)) if max_val > 0 else 0
    z0 = float(z[z0_idx])
    
    print(f"   Superficie agua: Z0 = {z0:.1f} mm (bin {z0_idx})")
    
    # Métricas relativas a la superficie
    z_rel = z - z0
    zmax = float(z_rel[max_idx])
    
    # R50: última posición >= 50%
    r50 = None
    if max_val > 0:
        idx_50 = np.where(profile >= 0.5 * max_val)[0]
        if len(idx_50) > 0:
            r50 = float(z_rel[idx_50[-1]])
    
    # FWHM en profundidad
    fwhm = None
    if max_val > 0:
        idx_fwhm = np.where(profile >= 0.5 * max_val)[0]
        if len(idx_fwhm) > 1:
            fwhm = float(z_rel[idx_fwhm[-1]] - z_rel[idx_fwhm[0]])
    
    print(f"\n✅ MÉTRICAS TG-51:")
    print(f"   Zmax: {zmax:.1f} mm  (ref: ~13-14 mm para 6 MeV)")
    if r50 is not None:
        print(f"   R50:  {r50:.1f} mm  (ref: ~26 mm para 6 MeV)")
    if fwhm is not None:
        print(f"   FWHM: {fwhm:.1f} mm")
    
    # PDD (Percentage Depth Dose)
    pdd = (profile / max_val * 100) if max_val > 0 else profile
    
    # Guardar resultados
    results = {
        'z0_mm': z0,
        'zmax_rel_mm': zmax,
        'r50_rel_mm': r50,
        'fwhm_mm': fwhm,
        'max_dose': max_val,
        'spacing_mm': spacing,
        'shape': list(arr.shape),
    }
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Guardar PDD
    pdd_data = np.column_stack([z_rel, pdd])
    np.savetxt(
        output_dir / 'pdd.csv',
        pdd_data,
        delimiter=',',
        header='z_rel_mm,pdd_percent',
        comments='',
        fmt='%.3f'
    )
    
    print(f"\n💾 Guardados:")
    print(f"   {output_dir / 'metrics.json'}")
    print(f"   {output_dir / 'pdd.csv'}")
    
    # Graficar si se solicita
    if args.plot:
        plot_pdd(z_rel, pdd, zmax, r50, output_dir)
    
    return 0


def plot_pdd(z_rel, pdd, zmax, r50, output_dir):
    """Genera gráfica del PDD."""
    
    print(f"\n📊 Generando gráfica...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Rango útil: 0 a 60mm
    mask = (z_rel >= 0) & (z_rel <= 60)
    
    ax.plot(z_rel[mask], pdd[mask], 'b-', linewidth=2, label='PDD')
    ax.axvline(zmax, color='red', linestyle='--', linewidth=1.5, label=f'Zmax = {zmax:.1f} mm')
    
    if r50 is not None:
        ax.axvline(r50, color='green', linestyle='--', linewidth=1.5, label=f'R50 = {r50:.1f} mm')
    
    ax.axhline(50, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Profundidad relativa (mm)', fontsize=12)
    ax.set_ylabel('PDD (%)', fontsize=12)
    ax.set_title('Percentage Depth Dose (PDD)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    
    plot_path = output_dir / 'pdd_plot.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Análisis de perfil de dosis (parametrizado para cluster)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Archivo MHD de dosis (e.g., output/dose_z_edep.mhd)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Directorio para resultados del análisis'
    )
    
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generar gráfica del PDD'
    )
    
    args = parser.parse_args()
    
    exit_code = analyze_dose_profile(args)
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
