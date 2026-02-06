#!/usr/bin/env python3
"""
Test MÍNIMO para diagnosticar si MIOpen funciona en Yuca.
Prueba 4 configuraciones con una conv3d trivial.
"""
import os
import time
import sys

# ============================================
# TEST 1: GPU sin MIOpen (lo que ya funciona)
# ============================================
print("=" * 60)
print("TEST 1: GPU + cudnn OFF (baseline que funciona)")
print("=" * 60)
os.environ["MIOPEN_DEBUG_DISABLE_FIND_DB"] = "1"

import torch
import torch.nn as nn

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {device}")
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

try:
    conv = nn.Conv3d(1, 8, 3, padding=1).to(device)
    x = torch.randn(1, 1, 32, 32, 32, device=device)
    
    # Warmup
    y = conv(x)
    torch.cuda.synchronize()
    
    # Benchmark
    t0 = time.time()
    for _ in range(50):
        y = conv(x)
    torch.cuda.synchronize()
    t1 = time.time()
    
    print(f"  ✅ FUNCIONA - 50 iters en {t1-t0:.3f}s ({(t1-t0)/50*1000:.1f} ms/iter)")
except Exception as e:
    print(f"  ❌ ERROR: {e}")

# ============================================
# TEST 2: GPU + cudnn ON (sin benchmark)
# ============================================
print("\n" + "=" * 60)
print("TEST 2: GPU + cudnn ON + benchmark OFF")
print("=" * 60)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False

try:
    conv = nn.Conv3d(1, 8, 3, padding=1).to(device)
    x = torch.randn(1, 1, 32, 32, 32, device=device)
    
    y = conv(x)
    torch.cuda.synchronize()
    
    t0 = time.time()
    for _ in range(50):
        y = conv(x)
    torch.cuda.synchronize()
    t1 = time.time()
    
    print(f"  ✅ FUNCIONA - 50 iters en {t1-t0:.3f}s ({(t1-t0)/50*1000:.1f} ms/iter)")
except Exception as e:
    print(f"  ❌ FALLA: {e}")

# ============================================
# TEST 3: GPU + cudnn ON + benchmark ON
# ============================================
print("\n" + "=" * 60)
print("TEST 3: GPU + cudnn ON + benchmark ON")
print("=" * 60)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

try:
    conv = nn.Conv3d(1, 8, 3, padding=1).to(device)
    x = torch.randn(1, 1, 32, 32, 32, device=device)
    
    print("  Ejecutando primera iter (benchmark busca algoritmo)...")
    y = conv(x)
    torch.cuda.synchronize()
    print("  Primera iter OK, midiendo...")
    
    t0 = time.time()
    for _ in range(50):
        y = conv(x)
    torch.cuda.synchronize()
    t1 = time.time()
    
    print(f"  ✅ FUNCIONA - 50 iters en {t1-t0:.3f}s ({(t1-t0)/50*1000:.1f} ms/iter)")
except Exception as e:
    print(f"  ❌ FALLA: {e}")

# ============================================
# TEST 4: GPU + cudnn ON + FIND_DB habilitado
# ============================================
print("\n" + "=" * 60)
print("TEST 4: GPU + cudnn ON + FIND_DB=0 (permitido)")
print("=" * 60)

os.environ["MIOPEN_DEBUG_DISABLE_FIND_DB"] = "0"
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False

try:
    conv = nn.Conv3d(1, 8, 3, padding=1).to(device)
    x = torch.randn(1, 1, 32, 32, 32, device=device)
    
    y = conv(x)
    torch.cuda.synchronize()
    
    t0 = time.time()
    for _ in range(50):
        y = conv(x)
    torch.cuda.synchronize()
    t1 = time.time()
    
    print(f"  ✅ FUNCIONA - 50 iters en {t1-t0:.3f}s ({(t1-t0)/50*1000:.1f} ms/iter)")
except Exception as e:
    print(f"  ❌ FALLA: {e}")

# ============================================
# RESUMEN
# ============================================
print("\n" + "=" * 60)
print("RESUMEN: Copia este output y pégalo")
print("=" * 60)
print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
print(f"ROCM: {torch.version.hip if hasattr(torch.version, 'hip') else 'N/A'}")
