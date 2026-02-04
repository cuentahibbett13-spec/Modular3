#!/usr/bin/env python3
"""
Inference - Predicción rápida en imágenes nuevas
Carga modelo entrenado y hace denoising de volúmenes ruidosos
"""

import torch
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from tqdm import tqdm

from model import create_model


class DoseInference:
    """Inference engine para denoising"""
    
    def __init__(self, model_path, device='cuda'):
        """
        Args:
            model_path: Ruta al checkpoint del modelo
            device: 'cuda' o 'cpu'
        """
        self.device = device
        
        # Cargar modelo
        self.model = create_model(device=device)
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        
        print(f"✅ Model loaded: {model_path}")
    
    def _load_mhd(self, path):
        """Cargar archivo MHD"""
        img = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        return arr, img
    
    def _normalize_volume(self, arr):
        """Normalizar volumen igual que training"""
        nonzero = arr[arr > 0]
        if len(nonzero) > 0:
            mean = nonzero.mean()
            std = nonzero.std() + 1e-8
            arr[arr > 0] = (arr[arr > 0] - mean) / std
        return arr
    
    def _denormalize_volume(self, arr, original_arr):
        """Desnormalizar volumen a escala original"""
        nonzero_orig = original_arr[original_arr > 0]
        if len(nonzero_orig) > 0:
            mean = nonzero_orig.mean()
            std = nonzero_orig.std() + 1e-8
            
            nonzero = arr[arr > 0]
            if len(nonzero) > 0:
                arr[arr > 0] = arr[arr > 0] * std + mean
        
        # Clamp a positivo
        arr[arr < 0] = 0
        
        return arr
    
    @torch.no_grad()
    def predict(self, noisy_path, output_path, batch_size=1):
        """
        Hacer predicción en volumen ruidoso
        
        Args:
            noisy_path: Ruta a imagen MHD ruidosa
            output_path: Ruta para guardar resultado denoised
            batch_size: Tamaño de batch (para volúmenes muy grandes)
        """
        print(f"\n📊 Processing: {noisy_path}")
        
        # Cargar
        noisy_arr, img_obj = self._load_mhd(noisy_path)
        original_arr = noisy_arr.copy()
        
        print(f"  Shape: {noisy_arr.shape}")
        print(f"  Range: [{noisy_arr.min():.3e}, {noisy_arr.max():.3e}]")
        
        # Normalizar
        noisy_arr = self._normalize_volume(noisy_arr)
        
        # Forward pass
        print("  Running inference...")
        
        noisy_tensor = torch.from_numpy(
            noisy_arr[np.newaxis, np.newaxis, ...]
        ).float().to(self.device)
        
        with torch.no_grad():
            clean_tensor = self.model(noisy_tensor)
        
        clean_arr = clean_tensor.cpu().numpy()[0, 0]
        
        # Desnormalizar
        clean_arr = self._denormalize_volume(clean_arr, original_arr)
        
        print(f"  Denoised range: [{clean_arr.min():.3e}, {clean_arr.max():.3e}]")
        
        # Guardar
        clean_img = sitk.GetImageFromArray(clean_arr)
        clean_img.CopyInformation(img_obj)
        sitk.WriteImage(clean_img, str(output_path))
        
        print(f"  ✅ Saved: {output_path}")
        
        # Estadísticas de denoising
        noisy_nonzero = original_arr[original_arr > 0]
        clean_nonzero = clean_arr[clean_arr > 0]
        
        noise_reduction = 1 - (clean_nonzero.std() / noisy_nonzero.std())
        print(f"  📈 Noise reduction: {noise_reduction*100:.1f}%")
        
        return clean_arr
    
    @torch.no_grad()
    def predict_batch(self, noisy_paths, output_dir='denoised', batch_size=1):
        """
        Hacer predicción en múltiples volúmenes
        
        Args:
            noisy_paths: Lista de rutas o patrón glob
            output_dir: Directorio de salida
            batch_size: Tamaño de batch
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Expandir paths
        if isinstance(noisy_paths, str):
            noisy_paths = list(Path('.').glob(noisy_paths))
        
        print(f"\n🔄 Batch processing {len(noisy_paths)} volumes...")
        print("="*60)
        
        for noisy_path in tqdm(noisy_paths):
            output_path = output_dir / f"denoised_{noisy_path.stem}.mhd"
            self.predict(noisy_path, output_path)
        
        print("\n✅ Batch processing complete!")


def main():
    """Ejemplo de inference"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Denoising inference')
    parser.add_argument('--model', required=True, help='Path to model checkpoint')
    parser.add_argument('--input', required=True, help='Input MHD file or glob pattern')
    parser.add_argument('--output', default='denoised', help='Output directory')
    parser.add_argument('--device', default='cuda', help='Device: cuda or cpu')
    
    args = parser.parse_args()
    
    # Verificar device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Inference
    inference = DoseInference(args.model, device=args.device)
    
    # Predicción
    input_path = Path(args.input)
    if input_path.is_file():
        # Single file
        output_path = Path(args.output) / f"denoised_{input_path.stem}.mhd"
        Path(args.output).mkdir(exist_ok=True)
        inference.predict(str(input_path), str(output_path))
    else:
        # Batch
        inference.predict_batch(str(args.input), output_dir=args.output)


if __name__ == '__main__':
    # Ejemplo sin argumentos
    print("Usage:")
    print("  python inference.py --model checkpoints/model_best.pth --input noisy.mhd")
    print("  python inference.py --model checkpoints/model_best.pth --input 'results/*/*.mhd'")
    print("\nOr import as module:")
    print("  from inference import DoseInference")
    print("  inf = DoseInference('model_best.pth')")
    print("  inf.predict('noisy.mhd', 'denoised.mhd')")
