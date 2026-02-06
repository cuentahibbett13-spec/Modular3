#!/usr/bin/env python3
"""
Test 2: ¿Se arregla MIOpen si ponemos TMPDIR ANTES de importar torch?
El error es: boost::filesystem::temp_directory_path: Not a directory
"""
import os
from pathlib import Path
import time

# ============================================
# FIJAR TMPDIR *ANTES* de importar torch
# ============================================
tmpdir = Path.home() / "miopen_tmp"
tmpdir.mkdir(parents=True, exist_ok=True)

os.environ["TMPDIR"] = str(tmpdir)
os.environ["TEMP"] = str(tmpdir)
os.environ["TMP"] = str(tmpdir)
os.environ["MIOPEN_USER_DB_PATH"] = str(tmpdir)
os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = str(tmpdir)

print(f"TMPDIR = {os.environ['TMPDIR']}")
print(f"Directorio existe: {tmpdir.exists()}")
print(f"Es escribible: {os.access(str(tmpdir), os.W_OK)}")

# AHORA importar torch
import torch
import torch.nn as nn

device = torch.device('cuda')
print(f"\nPyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================
# TEST A: cudnn ON (con TMPDIR arreglado)
# ============================================
print("\n" + "=" * 60)
print("TEST A: cudnn ON + TMPDIR arreglado")
print("=" * 60)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False

try:
    conv = nn.Conv3d(1, 8, 3, padding=1).to(device)
    x = torch.randn(1, 1, 32, 32, 32, device=device)
    
    y = conv(x)
    torch.cuda.synchronize()
    
    t0 = time.time()
    for _ in range(100):
        y = conv(x)
    torch.cuda.synchronize()
    t1 = time.time()
    
    print(f"  ✅ FUNCIONA - 100 iters en {t1-t0:.3f}s ({(t1-t0)/100*1000:.2f} ms/iter)")
except Exception as e:
    print(f"  ❌ FALLA: {e}")

# ============================================
# TEST B: cudnn ON + benchmark ON
# ============================================
print("\n" + "=" * 60)
print("TEST B: cudnn ON + benchmark ON + TMPDIR arreglado")
print("=" * 60)

torch.backends.cudnn.benchmark = True

try:
    conv = nn.Conv3d(1, 16, 3, padding=1).to(device)
    x = torch.randn(2, 1, 64, 64, 64, device=device)
    
    print("  Warmup (benchmark busca algoritmo)...")
    y = conv(x)
    torch.cuda.synchronize()
    print("  Warmup OK")
    
    t0 = time.time()
    for _ in range(100):
        y = conv(x)
    torch.cuda.synchronize()
    t1 = time.time()
    
    print(f"  ✅ FUNCIONA - 100 iters en {t1-t0:.3f}s ({(t1-t0)/100*1000:.2f} ms/iter)")
except Exception as e:
    print(f"  ❌ FALLA: {e}")

# ============================================
# TEST C: Comparación OFF vs ON
# ============================================
print("\n" + "=" * 60)
print("TEST C: Comparación velocidad OFF vs ON (conv3d grande)")
print("=" * 60)

# Tamaño más realista (similar al training)
x_big = torch.randn(1, 1, 128, 128, 128, device=device)

# OFF
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
conv_off = nn.Conv3d(1, 16, 3, padding=1).to(device)
y = conv_off(x_big)
torch.cuda.synchronize()

t0 = time.time()
for _ in range(20):
    y = conv_off(x_big)
torch.cuda.synchronize()
t_off = time.time() - t0
print(f"  cudnn OFF: 20 iters en {t_off:.3f}s ({t_off/20*1000:.1f} ms/iter)")

# ON
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
conv_on = nn.Conv3d(1, 16, 3, padding=1).to(device)
y = conv_on(x_big)
torch.cuda.synchronize()

t0 = time.time()
for _ in range(20):
    y = conv_on(x_big)
torch.cuda.synchronize()
t_on = time.time() - t0
print(f"  cudnn ON:  20 iters en {t_on:.3f}s ({t_on/20*1000:.1f} ms/iter)")

speedup = t_off / t_on if t_on > 0 else 0
print(f"\n  🏁 Speedup con MIOpen: {speedup:.1f}x")

print("\n" + "=" * 60)
print("FIN - Si Test A pasó, MIOpen funciona con TMPDIR arreglado")
print("=" * 60)
