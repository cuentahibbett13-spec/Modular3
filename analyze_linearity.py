#!/usr/bin/env python3
"""
Análisis de Linealidad: Input Max vs Pred Max.
Si es lineal con R²≈1, el modelo es un escalador de dosis confiable.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import SimpleITK as sitk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

# Desactivar MIOpen
torch.backends.cudnn.enabled = False

# ---- Configuración ----
# Buscar modelo: primero fullvol, luego v2
MODEL_PATHS = [
    Path("runs/denoising_fullvol/best.pt"),
    Path("runs/denoising_v2/best.pt"),
]

MODEL_PATH = None
for p in MODEL_PATHS:
    if p.exists():
        MODEL_PATH = p
        break

if MODEL_PATH is None:
    print(f"❌ No existe ningún modelo en:")
    for p in MODEL_PATHS:
        print(f"   - {p}")
    exit(1)
VAL_DIR = DATASET_ROOT / "val"
INPUT_LEVELS = ["input_1M", "input_2M", "input_5M", "input_10M"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🔧 Device: {DEVICE}")


def read_volume(mhd_path: Path) -> np.ndarray:
    """Lee archivo .mhd."""
    npy_path = mhd_path.with_suffix(".npy")
    if npy_path.exists():
        return np.load(str(npy_path)).astype(np.float32)
    img = sitk.ReadImage(str(mhd_path))
    return sitk.GetArrayFromImage(img).astype(np.float32)


# ---- Modelo UNet3D ----
def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.GroupNorm(8, out_ch),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.GroupNorm(8, out_ch),
        nn.ReLU(inplace=True),
    )


class UNet3D(nn.Module):
    def __init__(self, base_ch=16):
        super().__init__()
        self.enc1 = conv_block(1, base_ch)
        self.enc2 = conv_block(base_ch, base_ch*2)
        self.enc3 = conv_block(base_ch*2, base_ch*4)
        self.pool = nn.MaxPool3d(2)
        self.bottleneck = conv_block(base_ch*4, base_ch*8)
        self.up3 = nn.ConvTranspose3d(base_ch*8, base_ch*4, 2, stride=2)
        self.dec3 = conv_block(base_ch*8, base_ch*4)
        self.up2 = nn.ConvTranspose3d(base_ch*4, base_ch*2, 2, stride=2)
        self.dec2 = conv_block(base_ch*4, base_ch*2)
        self.up1 = nn.ConvTranspose3d(base_ch*2, base_ch, 2, stride=2)
        self.dec1 = conv_block(base_ch*2, base_ch)
        self.out = nn.Conv3d(base_ch, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.out(d1)


def main():
    print("=" * 70)
    print("ANÁLISIS DE LINEALIDAD: INPUT MAX vs PREDICCIÓN MAX")
    print("=" * 70)
    
    # Cargar modelo
    assert MODEL_PATH is not None, "❌ No existe modelo"
    model = UNet3D(base_ch=16).to(DEVICE)
    ckpt = torch.load(str(MODEL_PATH), map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"✅ Modelo cargado: {MODEL_PATH}")
    print(f"   Epoch: {ckpt['epoch']}, Val Loss: {ckpt['val_loss']:.6f}\n")
    
    # Recolectar datos
    input_maxs = []
    pred_maxs = []
    pair_names = []
    level_names = []
    
    n_targets = len(list(DATASET_ROOT.glob("target_*")))
    
    for pair_dir in sorted(VAL_DIR.glob("pair_*")):
        pair_num = int(pair_dir.name.split("_")[-1])
        target_idx = ((pair_num - 1) % n_targets) + 1
        target_mhd = DATASET_ROOT / f"target_{target_idx}" / "dose_edep.mhd"
        
        if not target_mhd.exists():
            continue
        
        # Leer target una vez
        tgt = read_volume(target_mhd)
        max_val = float(np.max(tgt))
        if max_val <= 0:
            continue
        
        for level in INPUT_LEVELS:
            input_mhd = pair_dir / f"{level}.mhd"
            if not input_mhd.exists():
                input_mhd = pair_dir / level / "dose_edep.mhd"
            
            if not input_mhd.exists():
                continue
            
            # Leer input
            inp = read_volume(input_mhd)
            inp_norm = inp / max_val
            tgt_norm = tgt / max_val
            
            # Inferencia
            inp_t = torch.from_numpy(inp_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                pred_t = model(inp_t)
            pred = pred_t.squeeze().cpu().numpy()
            pred = np.clip(pred, 0, 1)
            
            # Extraer máximos
            inp_max = float(np.max(inp_norm))
            pred_max = float(np.max(pred))
            tgt_max = float(np.max(tgt_norm))
            
            input_maxs.append(inp_max)
            pred_maxs.append(pred_max)
            pair_names.append(pair_dir.name)
            level_names.append(level)
            
            print(f"{pair_dir.name}/{level}: Input Max={inp_max:.4f}, Pred Max={pred_max:.4f}, Target Max={tgt_max:.4f}")
    
    # Análisis de regresión lineal
    print("\n" + "=" * 70)
    print("ANÁLISIS ESTADÍSTICO")
    print("=" * 70)
    
    input_maxs = np.array(input_maxs)
    pred_maxs = np.array(pred_maxs)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(input_maxs, pred_maxs)
    
    print(f"✅ Regresión Lineal: y = {slope:.4f}*x + {intercept:.6f}")
    print(f"   R² = {r_value**2:.6f} (1.0 = perfecta linealidad)")
    print(f"   Pendiente = {slope:.4f} (factor de amplificación)")
    print(f"   Intercepto = {intercept:.6f} (offset)")
    print(f"   p-value = {p_value:.2e} (significancia)")
    
    # Gráfica
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfica 1: Scatter + regresión
    ax1.scatter(input_maxs, pred_maxs, s=100, alpha=0.6, color='blue', edgecolors='black', linewidth=1.5)
    
    # Línea de regresión
    x_line = np.array([input_maxs.min(), input_maxs.max()])
    y_line = slope * x_line + intercept
    ax1.plot(x_line, y_line, 'r--', linewidth=2, label=f'y = {slope:.4f}x + {intercept:.6f}')
    
    # Línea ideal (slope=1, intercept=0) si fuera perfecto
    y_ideal = x_line
    ax1.plot(x_line, y_ideal, 'g:', linewidth=2, alpha=0.7, label='Ideal (y=x)')
    
    ax1.set_xlabel('Input Max (normalizado)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Predicción Max (normalizado)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Linealidad del Modelo\nR² = {r_value**2:.6f}', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Gráfica 2: Residuales
    y_pred_line = slope * input_maxs + intercept
    residuals = pred_maxs - y_pred_line
    
    ax2.scatter(input_maxs, residuals, s=100, alpha=0.6, color='orange', edgecolors='black', linewidth=1.5)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax2.fill_between([input_maxs.min(), input_maxs.max()], 
                      [-2*std_err, -2*std_err], 
                      [2*std_err, 2*std_err], 
                      alpha=0.2, color='green', label='±2 σ')
    
    ax2.set_xlabel('Input Max (normalizado)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Residuales', fontsize=12, fontweight='bold')
    ax2.set_title('Análisis de Residuales\n(Desviación de la línea ideal)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig("linearity_analysis.png", dpi=150, bbox_inches='tight')
    print(f"\n📊 Gráfica guardada: linearity_analysis.png")
    
    # Veredicto
    print("\n" + "=" * 70)
    print("VEREDICTO")
    print("=" * 70)
    
    if r_value**2 > 0.95:
        print("✅ MUY LINEAL: El modelo es un escalador de dosis EXCELENTE")
        print("   → Predicción confiable y proporcional al input")
    elif r_value**2 > 0.85:
        print("✅ LINEAL: El modelo es un buen escalador de dosis")
        print("   → Comportamiento predecible con algunos artefactos")
    elif r_value**2 > 0.70:
        print("⚠️  PARCIALMENTE LINEAL: Algunos niveles desviados")
        print("   → Mejoras posibles en arquitectura o pérdida")
    else:
        print("❌ NO LINEAL: El modelo tiene problemas de calibración")
        print("   → Revisar normalización o arquitectura")
    
    print(f"\n📈 Pendiente = {slope:.4f}")
    print(f"   (Ideal: ~0.3-0.5 si el modelo aprende bien la amplificación)")


if __name__ == "__main__":
    main()
