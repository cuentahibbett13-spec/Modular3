#!/usr/bin/env python3
"""
Debug gamma pass rate - entender por qué input tiene 0% incluso en input_10M
"""
import os
os.environ["MIOPEN_DEBUG_DISABLE_FIND_DB"] = "1"

import torch
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# CONFIG
# ============================================================================
BASE_CHANNELS = 16
MODEL_PATH = Path("runs/denoising_v2_residual/best_model.pt")
DATASET_ROOT = Path("dataset_pilot")
VAL_DIR = DATASET_ROOT / "val"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATCH_SIZE = 96
OVERLAP = 16

# ============================================================================
# 3D U-NET (COPIADO DE EVALUATE_MODEL.PY)
# ============================================================================
class ResidualUNet3D(torch.nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=16):
        super().__init__()
        self.enc1 = self._conv_block(in_channels, base_channels)
        self.pool1 = torch.nn.MaxPool3d(2)
        self.enc2 = self._conv_block(base_channels, base_channels * 2)
        self.pool2 = torch.nn.MaxPool3d(2)
        self.enc3 = self._conv_block(base_channels * 2, base_channels * 4)
        self.pool3 = torch.nn.MaxPool3d(2)
        self.bottleneck = self._conv_block(base_channels * 4, base_channels * 8)
        self.upconv3 = torch.nn.ConvTranspose3d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = self._conv_block(base_channels * 8, base_channels * 4)
        self.upconv2 = torch.nn.ConvTranspose3d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = self._conv_block(base_channels * 4, base_channels * 2)
        self.upconv1 = torch.nn.ConvTranspose3d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = self._conv_block(base_channels * 2, base_channels)
        self.final = torch.nn.Conv3d(base_channels, out_channels, 1)

    def _conv_block(self, in_ch, out_ch):
        return torch.nn.Sequential(
            torch.nn.Conv3d(in_ch, out_ch, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv3d(out_ch, out_ch, 3, padding=1),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.dec3(torch.cat([self.upconv3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upconv2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upconv1(d2), e1], dim=1))
        residual = self.final(d1)
        return x + residual  # RESIDUAL


# ============================================================================
# SLIDING WINDOW
# ============================================================================
def sliding_window_inference(model, volume, device, patch_size=96, overlap=16):
    """Copia de evaluate_model.py"""
    model.eval()
    z, y, x = volume.shape
    step = patch_size - overlap
    
    pad_z = (step - (z % step)) % step
    pad_y = (step - (y % step)) % step
    pad_x = (step - (x % step)) % step
    
    padded = np.pad(volume, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant')
    pz, py, px = padded.shape
    
    output = np.zeros_like(padded)
    weight_map = np.zeros_like(padded)
    
    w = np.ones((patch_size, patch_size, patch_size), dtype=np.float32)
    margin = overlap // 2
    if margin > 0:
        for i in range(margin):
            fade = (i + 1) / margin
            w[i, :, :] *= fade
            w[-(i+1), :, :] *= fade
            w[:, i, :] *= fade
            w[:, -(i+1), :] *= fade
            w[:, :, i] *= fade
            w[:, :, -(i+1)] *= fade
    
    positions = []
    for zs in range(0, pz, step):
        for ys in range(0, py, step):
            for xs in range(0, px, step):
                z_end = min(zs + patch_size, pz)
                y_end = min(ys + patch_size, py)
                x_end = min(xs + patch_size, px)
                
                if z_end - zs < patch_size:
                    zs = max(0, z_end - patch_size)
                if y_end - ys < patch_size:
                    ys = max(0, y_end - patch_size)
                if x_end - xs < patch_size:
                    xs = max(0, x_end - patch_size)
                
                positions.append((zs, ys, xs))
    
    seen = set()
    unique_positions = []
    for pos in positions:
        if pos not in seen:
            seen.add(pos)
            unique_positions.append(pos)
    positions = unique_positions
    
    with torch.no_grad():
        for zs, ys, xs in positions:
            patch = padded[zs:zs+patch_size, ys:ys+patch_size, xs:xs+patch_size]
            patch_t = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0).to(device)
            pred = model(patch_t)
            pred_np = pred.squeeze().cpu().numpy()
            output[zs:zs+patch_size, ys:ys+patch_size, xs:xs+patch_size] += pred_np * w
            weight_map[zs:zs+patch_size, ys:ys+patch_size, xs:xs+patch_size] += w
    
    weight_map = np.maximum(weight_map, 1e-8)
    output = output / weight_map
    return output[:z, :y, :x]


# ============================================================================
# GAMMA ANALYSIS - VERSIONES DIFERENTES
# ============================================================================
def calc_gamma_original(pred, target, mask, dose_tolerance=3.0):
    """Original: solo diferencia de dosis"""
    pred_masked = pred[mask]
    target_masked = target[mask]
    dose_diff = np.abs(pred_masked - target_masked)
    dose_gamma = dose_diff / (dose_tolerance / 100.0 * target_masked.max())
    pass_rate = np.sum(dose_gamma <= 1.0) / len(dose_gamma) * 100
    return pass_rate, dose_gamma


def calc_gamma_relaxed(pred, target, mask, dose_tolerance=5.0):
    """Versión relajada: 5% en vez de 3%"""
    pred_masked = pred[mask]
    target_masked = target[mask]
    dose_diff = np.abs(pred_masked - target_masked)
    dose_gamma = dose_diff / (dose_tolerance / 100.0 * target_masked.max())
    pass_rate = np.sum(dose_gamma <= 1.0) / len(dose_gamma) * 100
    return pass_rate, dose_gamma


def calc_gamma_local(pred, target, mask, dose_tolerance=3.0):
    """Versión local: 3% de la dosis local, no del máximo"""
    pred_masked = pred[mask]
    target_masked = target[mask]
    dose_diff = np.abs(pred_masked - target_masked)
    dose_gamma = dose_diff / (dose_tolerance / 100.0 * target_masked + 1e-10)
    pass_rate = np.sum(dose_gamma <= 1.0) / len(dose_gamma) * 100
    return pass_rate, dose_gamma


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "=" * 80)
    print("DEBUG: ¿Por qué el input tiene 0% gamma incluso input_10M?")
    print("=" * 80)
    
    # Cargar modelo
    print("\n[1] Cargando modelo...")
    model = ResidualUNet3D(base_channels=BASE_CHANNELS).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print(f"  ✓ Modelo cargado")
    
    # Usar el pair_021 como ejemplo
    pair_dir = VAL_DIR / "pair_021"
    target_idx = ((21 - 1) % 5) + 1
    target_mhd = DATASET_ROOT / f"target_{target_idx}" / "dose_edep.mhd"
    
    print(f"\n[2] Cargando pair_021 (target_{target_idx})...")
    target_vol = sitk.GetArrayFromImage(sitk.ReadImage(str(target_mhd))).astype(np.float32)
    max_dose = target_vol.max()
    print(f"  ✓ Target shape: {target_vol.shape}, max_dose: {max_dose:.4f}")
    
    # Cargar input_10M y hacer inferencia
    input_10M_path = pair_dir / "input_10M.mhd"
    input_10M = sitk.GetArrayFromImage(sitk.ReadImage(str(input_10M_path))).astype(np.float32)
    
    print(f"\n[3] Input_10M:")
    print(f"  Shape: {input_10M.shape}")
    print(f"  Max value: {input_10M.max():.4f}")
    print(f"  Mean value: {input_10M.mean():.4f}")
    print(f"  Stats - min/median/max: {np.min(input_10M):.4f} / {np.median(input_10M):.4f} / {np.max(input_10M):.4f}")
    
    print(f"\n[4] Target:")
    print(f"  Shape: {target_vol.shape}")
    print(f"  Max value: {max_dose:.4f}")
    print(f"  Mean value: {target_vol.mean():.4f}")
    print(f"  Stats - min/median/max: {np.min(target_vol):.4f} / {np.median(target_vol):.4f} / {np.max(target_vol):.4f}")
    
    # Diferencia directa
    diff = np.abs(input_10M - target_vol)
    print(f"\n[5] Diferencia directa (|input_10M - target|):")
    print(f"  Max: {diff.max():.4f}")
    print(f"  Mean: {diff.mean():.4f}")
    print(f"  Median: {np.median(diff):.4f}")
    print(f"  % de voxels dentro de 3% de max_dose: {np.sum(diff <= 0.03*max_dose) / diff.size * 100:.1f}%")
    print(f"  % de voxels dentro de 5% de max_dose: {np.sum(diff <= 0.05*max_dose) / diff.size * 100:.1f}%")
    
    # Ahora hacer inferencia con el modelo
    print(f"\n[6] Corriendo inferencia del modelo...")
    input_norm = input_10M / (max_dose + 1e-8)
    pred_norm = sliding_window_inference(model, input_norm, DEVICE, PATCH_SIZE, OVERLAP)
    pred_vol = pred_norm * max_dose
    
    print(f"  Predicción shape: {pred_vol.shape}")
    print(f"  Predicción max: {pred_vol.max():.4f}")
    print(f"  Predicción mean: {pred_vol.mean():.4f}")
    
    # Diferencia después de predicción
    diff_pred = np.abs(pred_vol - target_vol)
    print(f"\n[7] Diferencia predicción (|pred - target|):")
    print(f"  Max: {diff_pred.max():.4f}")
    print(f"  Mean: {diff_pred.mean():.4f}")
    print(f"  Median: {np.median(diff_pred):.4f}")
    print(f"  % de voxels dentro de 3% de max_dose: {np.sum(diff_pred <= 0.03*max_dose) / diff_pred.size * 100:.1f}%")
    print(f"  % de voxels dentro de 5% de max_dose: {np.sum(diff_pred <= 0.05*max_dose) / diff_pred.size * 100:.1f}%")
    
    # Ahora analizar la máscara y el gamma
    print(f"\n[8] Analizando máscaras y gamma...")
    
    # Máscara original: target > 0.10 * max_dose
    mask_10pct = target_vol > 0.10 * max_dose
    print(f"  Mask (target > 10% max_dose): {mask_10pct.sum():,} voxels ({mask_10pct.sum()/mask_10pct.size*100:.1f}%)")
    
    # Máscaras alternativas
    mask_5pct = target_vol > 0.05 * max_dose
    mask_1pct = target_vol > 0.01 * max_dose
    mask_0pct = target_vol > 0
    
    print(f"  Mask (target > 5% max_dose):  {mask_5pct.sum():,} voxels ({mask_5pct.sum()/mask_5pct.size*100:.1f}%)")
    print(f"  Mask (target > 1% max_dose):  {mask_1pct.sum():,} voxels ({mask_1pct.sum()/mask_1pct.size*100:.1f}%)")
    print(f"  Mask (target > 0):             {mask_0pct.sum():,} voxels ({mask_0pct.sum()/mask_0pct.size*100:.1f}%)")
    
    # Calcular gamma con diferentes configuraciones
    print(f"\n[9] GAMMA PARA INPUT_10M (diferentes configuraciones):")
    print(f"\nUsando mask 10% (original en evaluate_model.py):")
    if mask_10pct.sum() > 0:
        gamma_input_orig, gamma_vals = calc_gamma_original(input_10M, target_vol, mask_10pct, dose_tolerance=3.0)
        gamma_input_relaxed, _ = calc_gamma_relaxed(input_10M, target_vol, mask_10pct, dose_tolerance=5.0)
        gamma_input_local, _ = calc_gamma_local(input_10M, target_vol, mask_10pct, dose_tolerance=3.0)
        
        print(f"  - 3% (max dose based): {gamma_input_orig:.1f}%")
        print(f"  - 5% (max dose based): {gamma_input_relaxed:.1f}%")
        print(f"  - 3% (local dose):     {gamma_input_local:.1f}%")
        print(f"    Gamma values - min/median/max: {gamma_vals.min():.2f} / {np.median(gamma_vals):.2f} / {gamma_vals.max():.2f}")
    
    print(f"\nUsando mask 5%:")
    if mask_5pct.sum() > 0:
        gamma_input_orig, gamma_vals = calc_gamma_original(input_10M, target_vol, mask_5pct, dose_tolerance=3.0)
        gamma_input_relaxed, _ = calc_gamma_relaxed(input_10M, target_vol, mask_5pct, dose_tolerance=5.0)
        gamma_input_local, _ = calc_gamma_local(input_10M, target_vol, mask_5pct, dose_tolerance=3.0)
        
        print(f"  - 3% (max dose based): {gamma_input_orig:.1f}%")
        print(f"  - 5% (max dose based): {gamma_input_relaxed:.1f}%")
        print(f"  - 3% (local dose):     {gamma_input_local:.1f}%")
        print(f"    Gamma values - min/median/max: {gamma_vals.min():.2f} / {np.median(gamma_vals):.2f} / {gamma_vals.max():.2f}")
    
    print(f"\n[10] GAMMA PARA PREDICCIÓN (diferentes configuraciones):")
    print(f"\nUsando mask 10% (original en evaluate_model.py):")
    if mask_10pct.sum() > 0:
        gamma_pred_orig, gamma_vals = calc_gamma_original(pred_vol, target_vol, mask_10pct, dose_tolerance=3.0)
        gamma_pred_relaxed, _ = calc_gamma_relaxed(pred_vol, target_vol, mask_10pct, dose_tolerance=5.0)
        gamma_pred_local, _ = calc_gamma_local(pred_vol, target_vol, mask_10pct, dose_tolerance=3.0)
        
        print(f"  - 3% (max dose based): {gamma_pred_orig:.1f}%")
        print(f"  - 5% (max dose based): {gamma_pred_relaxed:.1f}%")
        print(f"  - 3% (local dose):     {gamma_pred_local:.1f}%")
        print(f"    Gamma values - min/median/max: {gamma_vals.min():.2f} / {np.median(gamma_vals):.2f} / {gamma_vals.max():.2f}")
    
    print(f"\nUsando mask 5%:")
    if mask_5pct.sum() > 0:
        gamma_pred_orig, gamma_vals = calc_gamma_original(pred_vol, target_vol, mask_5pct, dose_tolerance=3.0)
        gamma_pred_relaxed, _ = calc_gamma_relaxed(pred_vol, target_vol, mask_5pct, dose_tolerance=5.0)
        gamma_pred_local, _ = calc_gamma_local(pred_vol, target_vol, mask_5pct, dose_tolerance=3.0)
        
        print(f"  - 3% (max dose based): {gamma_pred_orig:.1f}%")
        print(f"  - 5% (max dose based): {gamma_pred_relaxed:.1f}%")
        print(f"  - 3% (local dose):     {gamma_pred_local:.1f}%")
        print(f"    Gamma values - min/median/max: {gamma_vals.min():.2f} / {np.median(gamma_vals):.2f} / {gamma_vals.max():.2f}")
    
    print("\n" + "=" * 80)
    print("CONCLUSIONES:")
    print("=" * 80)
    print("\nLa configuración actual usa:")
    print("  - Máscara: target > 10% max_dose")
    print("  - Tolerancia: 3% del max_dose")
    print("  - Cálculo: dose_gamma = |pred - target| / (0.03 * max_dose)")
    print("\nEsto es MUY estricto. El input ruidoso no puede pasar con esto.")
    print("Recomendación: Considerar")
    print("  - Máscara más inclusiva (5% o 1%)")
    print("  - O tolerancia local (3% de la dosis local, no del máximo)")
    print("  - O aumentar tolerancia a 5% del máximo")


if __name__ == '__main__':
    main()
