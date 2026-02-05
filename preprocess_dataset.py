#!/usr/bin/env python3
"""
Convierte todos los archivos MHD a NPY para I/O más rápida.
Ejecutar antes del entrenamiento.
"""

from pathlib import Path
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


def convert_mhd_to_npy(root_dir: Path):
    """Convierte todos los MHD en el dataset a NPY."""
    
    root_dir = Path(root_dir)
    count = 0
    
    # Targets
    for target_dir in sorted(root_dir.glob("target_*")):
        mhd_file = target_dir / "dose_edep.mhd"
        npy_file = target_dir / "dose_edep.npy"
        
        if mhd_file.exists() and not npy_file.exists():
            print(f"Converting {mhd_file}...")
            img = sitk.ReadImage(str(mhd_file))
            arr = sitk.GetArrayFromImage(img).astype(np.float32)
            np.save(str(npy_file), arr)
            count += 1
    
    # Train + Val pairs
    for pair_dir in sorted(root_dir.glob("train/pair_*")) + sorted(root_dir.glob("val/pair_*")):
        for level_dir in sorted(pair_dir.glob("input_*")):
            mhd_file = level_dir / "dose_edep.mhd"
            npy_file = level_dir / "dose_edep.npy"
            
            if mhd_file.exists() and not npy_file.exists():
                print(f"Converting {mhd_file}...")
                img = sitk.ReadImage(str(mhd_file))
                arr = sitk.GetArrayFromImage(img).astype(np.float32)
                np.save(str(npy_file), arr)
                count += 1
    
    print(f"\n✅ Convertidos {count} archivos a NPY")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    
    convert_mhd_to_npy(args.data_root)
