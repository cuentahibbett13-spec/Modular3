#!/usr/bin/env python3
"""
DataLoader para pares de dosis (noisy → clean)
Incluye augmentación geométrica y ruido Poisson opcional
"""

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from scipy import ndimage


class DoseDataset(Dataset):
    """
    Dataset de pares dosis noisy→clean para denoising
    
    Propiedades:
    - Augmentación: rotaciones 90°, flips en X/Y/Z
    - Ruido Poisson opcional (realista para imágenes médicas)
    - Normalización por volumen
    """
    
    def __init__(self, noisy_paths, clean_paths, augment=True, add_poisson=False, seed=None):
        """
        Args:
            noisy_paths: Lista de rutas a archivos MHD ruidosos
            clean_paths: Lista de rutas a archivos MHD limpios (targets)
            augment: Aplicar augmentación geométrica
            add_poisson: Agregar ruido Poisson sintético
            seed: Para reproducibilidad
        """
        assert len(noisy_paths) == len(clean_paths), "Misma cantidad de noisy y clean requerida"
        
        self.noisy_paths = [Path(p) for p in noisy_paths]
        self.clean_paths = [Path(p) for p in clean_paths]
        self.augment = augment
        self.add_poisson = add_poisson
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
        
        # Pre-cargar en memoria (opcional, descomenta si tenés RAM)
        # self.noisy_data = [self._load_mhd(p) for p in self.noisy_paths]
        # self.clean_data = [self._load_mhd(p) for p in self.clean_paths]
        # self.preloaded = True
        self.preloaded = False
    
    def _load_mhd(self, path):
        """Cargar archivo MHD como numpy array normalizado"""
        img = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        
        # Normalizar: (dose - mean) / std de valores no-cero
        nonzero = arr[arr > 0]
        if len(nonzero) > 0:
            mean = nonzero.mean()
            std = nonzero.std() + 1e-8
            arr[arr > 0] = (arr[arr > 0] - mean) / std
        
        return arr
    
    def _apply_poisson_noise(self, arr, scale=100):
        """
        Agregar ruido Poisson realista
        Típico en imágenes de detección (fotones, etc.)
        """
        # Escalar a fotones
        arr_photons = np.maximum(arr * scale, 0)
        
        # Poisson noise
        noisy = np.random.poisson(arr_photons).astype(np.float32) / scale
        
        # Re-normalizar
        nonzero = noisy[noisy > 0]
        if len(nonzero) > 0:
            noisy[noisy > 0] = (noisy[noisy > 0] - nonzero.mean()) / (nonzero.std() + 1e-8)
        
        return noisy
    
    def _apply_geometric_augmentation(self, arr):
        """
        Augmentación geométrica: rotaciones 90° + flips
        Físicamente válido (dosis es invariante a rotaciones)
        """
        # Número de rotaciones 90° (0, 1, 2, 3)
        k = np.random.randint(0, 4)
        arr = np.rot90(arr, k, axes=(1, 2))  # Rotar en plano XY
        
        # Flips aleatorios
        if np.random.rand() > 0.5:
            arr = np.flip(arr, axis=0)  # Flip en Z
        if np.random.rand() > 0.5:
            arr = np.flip(arr, axis=1)  # Flip en Y
        if np.random.rand() > 0.5:
            arr = np.flip(arr, axis=2)  # Flip en X
        
        return np.ascontiguousarray(arr)
    
    def __len__(self):
        return len(self.noisy_paths)
    
    def __getitem__(self, idx):
        """
        Retorna: (noisy, clean) como tensores [1, 125, 125, 125]
        """
        # Cargar
        if self.preloaded:
            noisy = self.noisy_data[idx].copy()
            clean = self.clean_data[idx].copy()
        else:
            noisy = self._load_mhd(self.noisy_paths[idx])
            clean = self._load_mhd(self.clean_paths[idx])
        
        # Augmentación (mismo transform en noisy y clean)
        if self.augment:
            aug = self._apply_geometric_augmentation(clean)
            # Aplicar las MISMAS transformaciones a noisy
            k = np.random.randint(0, 4)
            noisy = np.rot90(noisy, k, axes=(1, 2))
            clean = np.rot90(clean, k, axes=(1, 2))
            
            # Flips
            for axis in [0, 1, 2]:
                if np.random.rand() > 0.5:
                    noisy = np.flip(noisy, axis=axis)
                    clean = np.flip(clean, axis=axis)
            
            noisy = np.ascontiguousarray(noisy)
            clean = np.ascontiguousarray(clean)
        
        # Ruido Poisson opcional
        if self.add_poisson:
            clean = self._apply_poisson_noise(clean)
        
        # Convertir a tensores: [1, 125, 125, 125]
        noisy_tensor = torch.from_numpy(noisy[np.newaxis, ...]).float()
        clean_tensor = torch.from_numpy(clean[np.newaxis, ...]).float()
        
        return noisy_tensor, clean_tensor


def get_dataloaders(data_dir='results/iaea_final', batch_size=4, num_workers=0, 
                   train_pairs=None, val_pairs=None, augment=True, seed=42):
    """
    Crear DataLoaders para training y validation
    
    Args:
        data_dir: Directorio base con datasets
        batch_size: Tamaño de batch
        num_workers: Workers para carga
        train_pairs: Lista de tuplas (noisy_label, clean_label) para training
        val_pairs: Lista de tuplas para validation
        augment: Aplicar augmentación
        seed: Seed para reproducibilidad
    
    Returns:
        train_loader, val_loader
    """
    data_dir = Path(data_dir)
    # Convertir a path absoluto si es relativo
    if not data_dir.is_absolute():
        data_dir = Path(__file__).parent.parent / data_dir
    
    # Pares por defecto (recomendado)
    if train_pairs is None:
        train_pairs = [
            ('dose_50k', 'dose_5M'),
            ('dose_100k', 'dose_10M'),
            ('dose_500k', 'dose_full'),
        ]
    
    if val_pairs is None:
        val_pairs = [
            ('dose_1M', 'dose_full'),
        ]
    
    # Construir rutas
    train_noisy = [data_dir / f"{pair[0]}/dose_dose.mhd" for pair in train_pairs]
    train_clean = [data_dir / f"{pair[1]}/dose_dose.mhd" for pair in train_pairs]
    
    val_noisy = [data_dir / f"{pair[0]}/dose_dose.mhd" for pair in val_pairs]
    val_clean = [data_dir / f"{pair[1]}/dose_dose.mhd" for pair in val_pairs]
    
    # Datasets
    train_dataset = DoseDataset(
        train_noisy, train_clean,
        augment=augment,
        add_poisson=False,  # Cambiar a True para más variabilidad
        seed=seed
    )
    
    val_dataset = DoseDataset(
        val_noisy, val_clean,
        augment=False,  # Sin augmentación en validation
        add_poisson=False,
        seed=seed
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == '__main__':
    # Test
    print("Testing DoseDataset...")
    
    train_loader, val_loader = get_dataloaders(
        batch_size=2,
        augment=True,
        seed=42
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Cargar un batch
    noisy_batch, clean_batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Noisy: {noisy_batch.shape}")
    print(f"  Clean: {clean_batch.shape}")
    print(f"  Dtype: {noisy_batch.dtype}")
    
    print("\n✅ Dataset funcionando correctamente")
