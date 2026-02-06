#!/usr/bin/env python3
"""
Visualización comparativa de predicciones para todos los niveles de input (1M, 2M, 5M, 10M).
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import SimpleITK as sitk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Desactivar MIOpen
torch.backends.cudnn.enabled = False

# ---- Configuración ----
MODEL_PATH = Path("runs/denoising_v2/best.pt")
DATASET_ROOT = Path("dataset_pilot")
VAL_DIR = DATASET_ROOT / "val"
INPUT_LEVELS = ["input_1M", "input_2M", "input_5M", "input_10M"]
PATCH_SIZE = (64, 64, 64)  # Mismo que se usó en training actual
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🔧 Device: {DEVICE}")


def read_volume(mhd_path: Path) -> np.ndarray:
    """Lee archivo .mhd."""
    npy_path = mhd_path.with_suffix(".npy")
    if npy_path.exists():
        return np.load(str(npy_path)).astype(np.float32)
    img = sitk.ReadImage(str(mhd_path))
    return sitk.GetArrayFromImage(img).astype(np.float32)


def center_crop(vol, patch_size):
    """Crop central."""
    z, y, x = vol.shape
    pz, py, px = patch_size
    sz = max((z - pz) // 2, 0)
    sy = max((y - py) // 2, 0)
    sx = max((x - px) // 2, 0)
    return vol[sz:sz+pz, sy:sy+py, sx:sx+px]


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
    def __init__(self, base_ch=32):
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
    print("PREDICCIÓN PARA TODOS LOS NIVELES DE INPUT (1M, 2M, 5M, 10M)")
    print("=" * 70)
    
    # Cargar modelo
    assert MODEL_PATH.exists(), f"❌ No existe: {MODEL_PATH}"
    model = UNet3D(base_ch=32).to(DEVICE)
    ckpt = torch.load(str(MODEL_PATH), map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"✅ Modelo cargado (val_loss: {ckpt['val_loss']:.6f})\n")
    
    # Usar pair_021
    pair_021 = VAL_DIR / "pair_021"
    target_mhd = DATASET_ROOT / "target_1" / "dose_edep.mhd"
    
    # Leer target
    tgt = read_volume(target_mhd)
    max_val = float(np.max(tgt))
    tgt_norm = tgt / max_val
    tgt_crop = center_crop(tgt_norm, PATCH_SIZE)
    
    # Slice central para visualización
    slice_idx = PATCH_SIZE[0] // 2
    tgt_slice = tgt_crop[slice_idx, :, :]
    
    # Crear figura con 1 row target + 4 rows inputs
    fig, axes = plt.subplots(5, 3, figsize=(15, 20))
    fig.suptitle(
        f"Comparativa: Inputs (1M, 2M, 5M, 10M) vs Target (29.4M) vs Predicciones\nCorte Axial (Z={slice_idx})",
        fontsize=16, fontweight='bold'
    )
    
    cmap = 'hot'
    
    # Fila 0: Target (igual para todos)
    for col in range(3):
        if col == 0:
            ax = axes[0, col]
            im = ax.imshow(tgt_slice, cmap=cmap, origin='lower')
            ax.set_title("Target (29.4M)\n[Ground Truth]", fontweight='bold', color='green')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            axes[0, col].axis('off')
    
    # Filas 1-4: Input + Predicción
    results_table = []
    
    for row, level in enumerate(INPUT_LEVELS, start=1):
        # Input path
        input_mhd = pair_021 / f"{level}.mhd"
        if not input_mhd.exists():
            input_mhd = pair_021 / level / "dose_edep.mhd"
        
        if not input_mhd.exists():
            print(f"❌ No existe: {level}")
            # Llenar fila con mensajes de error
            for col in range(3):
                ax = axes[row, col]
                ax.text(0.5, 0.5, f"Archivo no encontrado:\n{level}", 
                       ha='center', va='center', fontsize=10, color='red')
                ax.axis('off')
            continue
        
        print(f"📂 {level}...")
        
        # Leer input
        inp = read_volume(input_mhd)
        inp_norm = inp / max_val
        inp_crop = center_crop(inp_norm, PATCH_SIZE)
        inp_slice = inp_crop[slice_idx, :, :]
        
        # Predicción
        inp_t = torch.from_numpy(inp_crop).unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred_t = model(inp_t)
        pred = pred_t.squeeze().cpu().numpy()
        pred = np.clip(pred, 0, 1)
        pred_slice = pred[slice_idx, :, :]
        
        # Métricas
        mae = np.mean(np.abs(tgt_crop - pred))
        mse = np.mean((tgt_crop - pred) ** 2)
        
        # Input range
        inp_max = np.max(inp_norm)
        pred_max = np.max(pred)
        
        print(f"   Input range: [{inp_norm.min():.6f}, {inp_max:.6f}]")
        print(f"   Pred range:  [{pred.min():.6f}, {pred_max:.6f}]")
        print(f"   MAE: {mae:.6f}, MSE: {mse:.6f}")
        
        results_table.append({
            'level': level,
            'inp_max': inp_max,
            'pred_max': pred_max,
            'mae': mae,
            'mse': mse
        })
        
        # Col 0: Input
        ax = axes[row, 0]
        im = ax.imshow(inp_slice, cmap=cmap, origin='lower')
        ax.set_title(f'{level}\n(Entrada Ruidosa)', fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Col 1: Predicción
        ax = axes[row, 1]
        im = ax.imshow(pred_slice, cmap=cmap, origin='lower')
        ax.set_title(f'{level}\n(Predicción)', fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Col 2: Diferencia (|Target - Pred|)
        diff = np.abs(tgt_slice - pred_slice)
        ax = axes[row, 2]
        im = ax.imshow(diff, cmap='viridis', origin='lower')
        ax.set_title(f'{level}\n|Target - Pred|', fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig("eval_all_levels.png", dpi=150, bbox_inches='tight')
    print(f"\n✅ Guardado: eval_all_levels.png")
    
    # Tabla resumen
    print("\n" + "=" * 100)
    print("📊 RESUMEN DE RESULTADOS")
    print("=" * 100)
    print(f"{'Nivel':<15} {'Input Max':<15} {'Pred Max':<15} {'MAE':<15} {'MSE':<15}")
    print("-" * 100)
    for r in results_table:
        print(f"{r['level']:<15} {r['inp_max']:<15.6f} {r['pred_max']:<15.6f} {r['mae']:<15.6f} {r['mse']:<15.6f}")
    print("=" * 100)
    
    # Análisis
    print("\n📋 ANÁLISIS:")
    print("   - ¿Pred Max aumenta con nivel de input? (esperado: sí)")
    print("   - ¿MAE es consistente entre niveles? (esperado: debería mejorar con más input)")
    print("   - ¿Predicción tiene estructura o es plana?")


if __name__ == "__main__":
    main()
