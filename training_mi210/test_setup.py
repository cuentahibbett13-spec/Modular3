#!/usr/bin/env python3
"""
Script de prueba rápida para verificar que todo funciona en el cluster
Ejecutar ANTES de lanzar el entrenamiento completo
"""

import sys
import torch
import numpy as np
from pathlib import Path

def check_pytorch_rocm():
    """Verificar PyTorch + ROCm"""
    print("\n" + "="*60)
    print("1. VERIFICANDO PYTORCH + ROCM")
    print("="*60)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA/ROCm available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        
        props = torch.cuda.get_device_properties(0)
        print(f"Total VRAM: {props.total_memory / 1e9:.1f} GB")
        print(f"Compute capability: {props.major}.{props.minor}")
        print("✅ ROCm funcionando correctamente")
    else:
        print("❌ ROCm NO detectado")
        print("Verificar:")
        print("  - ROCm instalado: rocm-smi")
        print("  - PyTorch con ROCm: pip install torch --index-url https://download.pytorch.org/whl/rocm5.7")
        sys.exit(1)

def check_dependencies():
    """Verificar dependencias"""
    print("\n" + "="*60)
    print("2. VERIFICANDO DEPENDENCIAS")
    print("="*60)
    
    try:
        import SimpleITK as sitk
        print(f"✅ SimpleITK {sitk.Version.VersionString()}")
    except ImportError:
        print("❌ SimpleITK no instalado: pip install SimpleITK")
        sys.exit(1)
    
    try:
        import tqdm
        print(f"✅ tqdm {tqdm.__version__}")
    except ImportError:
        print("❌ tqdm no instalado: pip install tqdm")
        sys.exit(1)
    
    try:
        import scipy
        print(f"✅ scipy {scipy.__version__}")
    except ImportError:
        print("❌ scipy no instalado: pip install scipy")
        sys.exit(1)

def check_data():
    """Verificar datasets"""
    print("\n" + "="*60)
    print("3. VERIFICANDO DATASETS")
    print("="*60)
    
    data_dir = Path(__file__).parent.parent / 'results' / 'iaea_final'
    
    if not data_dir.exists():
        print(f"❌ Directorio no encontrado: {data_dir}")
        print("Copiar datos con: rsync -avz results/ usuario@cluster:/ruta/Modular3/results/")
        sys.exit(1)
    
    required_datasets = ['dose_50k', 'dose_100k', 'dose_500k', 'dose_1M', 
                        'dose_2M', 'dose_5M', 'dose_10M', 'dose_full']
    
    for ds in required_datasets:
        ds_path = data_dir / ds / 'dose_dose.mhd'
        if ds_path.exists():
            print(f"✅ {ds}")
        else:
            print(f"❌ {ds} no encontrado")
            sys.exit(1)

def test_model_forward():
    """Test de forward pass"""
    print("\n" + "="*60)
    print("4. TEST FORWARD PASS")
    print("="*60)
    
    try:
        from model import UNet3D
        
        device = 'cuda'
        model = UNet3D(in_channels=1, out_channels=1).to(device)
        
        # Test input
        x = torch.randn(2, 1, 125, 125, 125).to(device)
        
        # Forward
        with torch.no_grad():
            y = model(x)
        
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {y.shape}")
        print(f"VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        assert y.shape == x.shape, "Output shape mismatch"
        print("✅ Forward pass OK")
        
    except Exception as e:
        print(f"❌ Error en forward pass: {e}")
        sys.exit(1)

def test_mixed_precision():
    """Test mixed precision (FP16)"""
    print("\n" + "="*60)
    print("5. TEST MIXED PRECISION (FP16)")
    print("="*60)
    
    try:
        from model import UNet3D
        
        device = 'cuda'
        model = UNet3D(in_channels=1, out_channels=1).to(device)
        
        x = torch.randn(4, 1, 125, 125, 125).to(device)
        
        # Test con autocast
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                y = model(x)
        
        print(f"Batch size: 4")
        print(f"VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print("✅ Mixed precision OK")
        
    except Exception as e:
        print(f"❌ Error en mixed precision: {e}")
        sys.exit(1)

def test_large_batch():
    """Test batch size 16 (MI210)"""
    print("\n" + "="*60)
    print("6. TEST BATCH SIZE 16 (MI210)")
    print("="*60)
    
    try:
        from model import UNet3D
        
        device = 'cuda'
        model = UNet3D(in_channels=1, out_channels=1).to(device)
        
        x = torch.randn(16, 1, 125, 125, 125).to(device)
        
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                y = model(x)
        
        vram_gb = torch.cuda.memory_allocated() / 1e9
        print(f"Batch size: 16")
        print(f"VRAM used: {vram_gb:.2f} GB / 64 GB")
        print(f"VRAM usage: {vram_gb/64*100:.1f}%")
        
        if vram_gb > 50:
            print("⚠️  VRAM muy alta - considerar batch_size=8")
        else:
            print("✅ Batch size 16 OK para MI210")
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("❌ Out of memory con batch=16")
            print("Reducir batch_size a 8 en train.py")
        else:
            raise
        sys.exit(1)

def main():
    print("\n" + "="*60)
    print("🔍 VERIFICACIÓN PRE-ENTRENAMIENTO (MI210)")
    print("="*60)
    
    check_pytorch_rocm()
    check_dependencies()
    check_data()
    test_model_forward()
    test_mixed_precision()
    test_large_batch()
    
    print("\n" + "="*60)
    print("✅ TODAS LAS VERIFICACIONES PASARON")
    print("="*60)
    print("\nPuedes iniciar el entrenamiento con:")
    print("  ./start_training_mi210.sh")
    print()

if __name__ == '__main__':
    main()
