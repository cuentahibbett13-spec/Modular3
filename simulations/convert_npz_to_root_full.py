#!/usr/bin/env python3
"""
Convertir NPZ IAEA a ROOT (versión completa o limitada).

Uso:
    # Completo (todos los electrones E>5.5 MeV)
    python convert_npz_to_root_full.py --full
    
    # Solo 500k (para tests)
    python convert_npz_to_root_full.py --n-particles 500000
"""

import numpy as np
from pathlib import Path
import uproot
import argparse

def convert_npz_to_root(npz_file, root_file, n_particles=None, min_energy=5.5):
    """
    Convierte NPZ a ROOT con filtros opcionales.
    
    Args:
        npz_file: Path del archivo NPZ de entrada
        root_file: Path del archivo ROOT de salida
        n_particles: Número máximo de partículas (None = todas)
        min_energy: Energía mínima en MeV
    """
    print("="*70)
    print("CONVERSIÓN NPZ → ROOT PARA PHASESPACESOURCE")
    print("="*70)
    
    # Cargar NPZ
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
    
    print(f"✅ Cargados {len(energy):,} partículas del NPZ")
    print(f"   Energía promedio: {energy.mean():.3f} MeV")
    print(f"   PDG types: {np.unique(pdg)}")
    
    # Filtrar electrones con energía mínima
    print(f"\n🔍 Filtrando: Electrones (PDG=11) con E > {min_energy} MeV")
    mask = (pdg == 11) & (energy > min_energy)
    mask_idx = np.where(mask)[0]
    
    print(f"   Encontrados: {len(mask_idx):,} electrones")
    
    # Limitar número si se especifica
    if n_particles is not None and len(mask_idx) > n_particles:
        print(f"   Limitando a: {n_particles:,} partículas")
        mask_idx = mask_idx[:n_particles]
    
    n_selected = len(mask_idx)
    print(f"✅ Seleccionados para conversión: {n_selected:,} electrones")
    
    # Extraer datos filtrados
    x_filtered = pos_x[mask_idx].astype(np.float32)
    y_filtered = pos_y[mask_idx].astype(np.float32)
    z_filtered = pos_z[mask_idx].astype(np.float32)
    dx_filtered = dir_u[mask_idx].astype(np.float32)
    dy_filtered = dir_v[mask_idx].astype(np.float32)
    dz_filtered = dir_w[mask_idx].astype(np.float32)
    e_filtered = energy[mask_idx].astype(np.float32)
    w_filtered = weight[mask_idx].astype(np.float32)
    pid_filtered = pdg[mask_idx].astype(np.int32)
    
    # Estadísticas finales
    print(f"\n📊 Estadísticas del dataset filtrado:")
    print(f"   Posición X: [{x_filtered.min():.2f}, {x_filtered.max():.2f}] cm")
    print(f"   Posición Y: [{y_filtered.min():.2f}, {y_filtered.max():.2f}] cm")
    print(f"   Posición Z: {z_filtered[0]:.2f} cm (constante)")
    print(f"   Energía: {e_filtered.mean():.3f} ± {e_filtered.std():.3f} MeV")
    print(f"   Peso: {w_filtered.mean():.6f} ± {w_filtered.std():.6f}")
    
    # Crear ROOT file con TTree
    print(f"\n💾 Guardando ROOT TTree: {root_file}")
    
    data_dict = {
        'x': x_filtered,
        'y': y_filtered,
        'z': z_filtered,
        'dx': dx_filtered,
        'dy': dy_filtered,
        'dz': dz_filtered,
        'E': e_filtered,
        'w': w_filtered,
        'pid': pid_filtered
    }
    
    with uproot.recreate(root_file) as file:
        file['phsp'] = data_dict
    
    # Verificar archivo creado
    file_size_mb = Path(root_file).stat().st_size / (1024**2)
    print(f"✅ ROOT file creado: {file_size_mb:.1f} MB")
    
    # Verificar integridad
    print(f"\n🔍 Verificando integridad...")
    with uproot.open(root_file) as f:
        tree = f['phsp']
        entries = tree.num_entries
        branches = tree.keys()
        print(f"   Entradas: {entries:,}")
        print(f"   Branches: {branches}")
        
        if entries == n_selected:
            print(f"✅ Verificación exitosa: {entries:,} entradas")
        else:
            print(f"⚠️  Warning: Esperadas {n_selected:,}, encontradas {entries:,}")
    
    print("\n" + "="*70)
    print("✅ CONVERSIÓN COMPLETADA")
    print("="*70)
    return root_file


def main():
    parser = argparse.ArgumentParser(
        description='Convertir NPZ IAEA a ROOT para PhaseSpaceSource',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Test con 500k partículas
  python convert_npz_to_root_full.py --n-particles 500000 --output phsp_500k.root
  
  # Conversión completa (todos los electrones E>5.5 MeV)
  python convert_npz_to_root_full.py --full --output phsp_full.root
  
  # Cambiar energía mínima
  python convert_npz_to_root_full.py --full --min-energy 6.0
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='data/IAEA/Varian_Clinac_2100CD_6MeV_15x15.npz',
        help='Archivo NPZ de entrada (default: Varian_Clinac_2100CD_6MeV_15x15.npz)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/IAEA/phsp_converted.root',
        help='Archivo ROOT de salida (default: phsp_converted.root)'
    )
    
    parser.add_argument(
        '--n-particles',
        type=int,
        default=None,
        help='Número máximo de partículas (default: todas las que pasen el filtro)'
    )
    
    parser.add_argument(
        '--full',
        action='store_true',
        help='Convertir todas las partículas (equivalente a --n-particles None)'
    )
    
    parser.add_argument(
        '--min-energy',
        type=float,
        default=5.5,
        help='Energía mínima en MeV para filtrar electrones (default: 5.5)'
    )
    
    args = parser.parse_args()
    
    # Si --full, ignorar n-particles
    if args.full:
        n_particles = None
        print("Modo COMPLETO: Convertir todas las partículas que pasen el filtro")
    else:
        n_particles = args.n_particles
    
    # Verificar que existe el archivo de entrada
    if not Path(args.input).exists():
        print(f"❌ ERROR: No se encuentra {args.input}")
        print("\nVerificar que el archivo NPZ existe.")
        print("Si necesitas generarlo desde IAEA, ejecutar:")
        print("  python simulations/convert_iaea_corrected.py")
        return 1
    
    # Crear directorio de salida si no existe
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Ejecutar conversión
    try:
        convert_npz_to_root(
            npz_file=args.input,
            root_file=args.output,
            n_particles=n_particles,
            min_energy=args.min_energy
        )
        return 0
    except Exception as e:
        print(f"\n❌ ERROR durante conversión: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
