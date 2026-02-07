#!/usr/bin/env python3
"""
Visualización 3D de Input vs Predicción vs GT
Genera isosurfaces en 3D para comparar volúmenes
"""
import os
os.environ["MIOPEN_DEBUG_DISABLE_FIND_DB"] = "1"

import torch
import torch.nn as nn
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

torch.backends.cudnn.enabled = False

# ============================================================================
# CONFIG
# ============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_CHANNELS = 16
MODEL_PATH = Path("runs/denoising_v2_residual/best_model.pt")
DATASET_ROOT = Path("dataset_pilot")
VAL_DIR = DATASET_ROOT / "val"
OUTPUT_DIR = Path("runs/denoising_v2_residual/visualizations_3d")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PAIR_TO_VISUALIZE = "pair_021"  # Cambiar según necesidad
INPUT_LEVEL = "input_10M"        # input_1M, input_2M, input_5M, input_10M

print(f"✓ Device: {DEVICE}")
print(f"✓ Visualizando: {PAIR_TO_VISUALIZE} | {INPUT_LEVEL}")

# ============================================================================
# 3D U-NET RESIDUAL
# ============================================================================
class ResidualUNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super().__init__()
        self.enc1 = self._conv_block(in_channels, base_channels)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = self._conv_block(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = self._conv_block(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool3d(2)
        self.bottleneck = self._conv_block(base_channels * 4, base_channels * 8)
        self.upconv3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = self._conv_block(base_channels * 8, base_channels * 4)
        self.upconv2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = self._conv_block(base_channels * 4, base_channels * 2)
        self.upconv1 = nn.ConvTranspose3d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = self._conv_block(base_channels * 2, base_channels)
        self.final = nn.Conv3d(base_channels, out_channels, 1)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
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
        return x + residual

# ============================================================================
# CARGA DE DATOS
# ============================================================================
def load_volumes():
    """Carga input, target y genera predicción"""
    # Encontrar archivos
    pair_dir = VAL_DIR / PAIR_TO_VISUALIZE
    input_path = pair_dir / f"{INPUT_LEVEL}.mhd"
    if not input_path.exists():
        input_path = pair_dir / INPUT_LEVEL / "dose_edep.mhd"
    
    # Determinar target
    pair_num = int(PAIR_TO_VISUALIZE.split("_")[-1])
    n_targets = len(list(DATASET_ROOT.glob("target_*")))
    target_idx = ((pair_num - 1) % n_targets) + 1
    target_path = DATASET_ROOT / f"target_{target_idx}" / "dose_edep.mhd"
    
    print(f"  Loading: {input_path}")
    print(f"  Target: {target_path}")
    
    input_vol = sitk.GetArrayFromImage(sitk.ReadImage(str(input_path))).astype(np.float32)
    target_vol = sitk.GetArrayFromImage(sitk.ReadImage(str(target_path))).astype(np.float32)
    
    # Cargar modelo
    model = ResidualUNet3D(base_channels=BASE_CHANNELS).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    # Inferencia con sliding window para no saturar GPU
    max_dose = target_vol.max()
    input_norm = input_vol / (max_dose + 1e-8)
    
    # Sliding window inference
    z, y, x = input_vol.shape
    patch_size = 96
    stride = 48
    
    pred_norm = np.zeros_like(input_norm)
    count_map = np.zeros_like(input_norm)
    
    print(f"  Inferencia con sliding window (patch={patch_size}, stride={stride})...")
    for z_start in range(0, z, stride):
        for y_start in range(0, y, stride):
            for x_start in range(0, x, stride):
                z_end = min(z_start + patch_size, z)
                y_end = min(y_start + patch_size, y)
                x_end = min(x_start + patch_size, x)
                
                # Ajustar si está en borde
                if z_end - z_start < patch_size:
                    z_start = max(0, z_end - patch_size)
                if y_end - y_start < patch_size:
                    y_start = max(0, y_end - patch_size)
                if x_end - x_start < patch_size:
                    x_start = max(0, x_end - patch_size)
                
                z_end = z_start + patch_size
                y_end = y_start + patch_size
                x_end = x_start + patch_size
                
                patch = input_norm[z_start:z_end, y_start:y_end, x_start:x_end]
                patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    pred_patch = model(patch_tensor).squeeze().cpu().numpy()
                
                pred_norm[z_start:z_end, y_start:y_end, x_start:x_end] += pred_patch
                count_map[z_start:z_end, y_start:y_end, x_start:x_end] += 1
    
    # Normalizar por count map
    pred_norm = pred_norm / (count_map + 1e-8)
    pred_vol = pred_norm * max_dose
    
    return input_vol, pred_vol, target_vol

# ============================================================================
# VISUALIZACIÓN 3D
# ============================================================================
def plot_3d_isosurface(volume, threshold_pct=0.5, title="Volume", color='red', alpha=0.3):
    """Genera isosurface 3D para un volumen"""
    max_val = volume.max()
    threshold = threshold_pct * max_val
    
    # Generar isosurface usando marching cubes
    verts, faces, _, _ = measure.marching_cubes(volume, level=threshold, spacing=(1, 1, 1))
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Crear mesh
    mesh = Poly3DCollection(verts[faces], alpha=alpha, edgecolor='none')
    mesh.set_facecolor(color)
    ax.add_collection3d(mesh)
    
    # Límites
    ax.set_xlim(0, volume.shape[0])
    ax.set_ylim(0, volume.shape[1])
    ax.set_zlim(0, volume.shape[2])
    
    ax.set_xlabel('Z')
    ax.set_ylabel('Y')
    ax.set_zlabel('X')
    ax.set_title(f"{title} (isosurface @ {threshold_pct*100}% max)")
    
    return fig

def plot_3d_comparison_complete(input_vol, pred_vol, target_vol, threshold_pct=0.5):
    """Genera comparación 3D con 6 subplots: Input, Pred, GT, Input-GT, Pred-GT, Error%"""
    fig = plt.figure(figsize=(24, 8))
    
    max_dose = target_vol.max()
    threshold = threshold_pct * max_dose
    
    # 1. Input
    ax1 = fig.add_subplot(161, projection='3d')
    plot_single_isosurface(ax1, input_vol, threshold_pct, "Input (10M)", 'red')
    
    # 2. Predicción
    ax2 = fig.add_subplot(162, projection='3d')
    plot_single_isosurface(ax2, pred_vol, threshold_pct, "Predicción (U-Net)", 'blue')
    
    # 3. Ground Truth
    ax3 = fig.add_subplot(163, projection='3d')
    plot_single_isosurface(ax3, target_vol, threshold_pct, "Ground Truth (29.4M)", 'green')
    
    # 4. Input - GT (diferencia)
    diff_input_gt = np.abs(input_vol - target_vol)
    ax4 = fig.add_subplot(164, projection='3d')
    plot_single_isosurface(ax4, diff_input_gt, 0.3, "Input - GT", 'orange')
    
    # 5. Pred - GT (diferencia)
    diff_pred_gt = np.abs(pred_vol - target_vol)
    ax5 = fig.add_subplot(165, projection='3d')
    plot_single_isosurface(ax5, diff_pred_gt, 0.3, "Pred - GT", 'purple')
    
    # 6. Error Relativo (%)
    error_pct = np.zeros_like(target_vol, dtype=np.float32)
    mask = target_vol > 0.01 * max_dose
    error_pct[mask] = 100.0 * np.abs(pred_vol[mask] - target_vol[mask]) / (target_vol[mask] + 1e-8)
    ax6 = fig.add_subplot(166, projection='3d')
    plot_single_isosurface(ax6, error_pct, 5.0, "Error % (Pred vs GT)", 'brown')
    
    plt.tight_layout()
    return fig

def plot_multiple_2d_slices(input_vol, pred_vol, target_vol, z_slices):
    """Muestra cortes 2D lado a lado para detectar desfases dimensionales"""
    n_slices = len(z_slices)
    fig, axes = plt.subplots(n_slices, 3, figsize=(15, 5*n_slices))
    
    if n_slices == 1:
        axes = axes.reshape(1, -1)
    
    max_dose = target_vol.max()
    
    for i, z in enumerate(z_slices):
        if z >= input_vol.shape[0]:
            continue
        
        # Input
        im0 = axes[i, 0].imshow(input_vol[z, :, :], cmap='hot', vmin=0, vmax=max_dose)
        axes[i, 0].set_title(f'Input @ z={z}')
        axes[i, 0].set_ylabel('Y')
        plt.colorbar(im0, ax=axes[i, 0])
        
        # Predicción
        im1 = axes[i, 1].imshow(pred_vol[z, :, :], cmap='hot', vmin=0, vmax=max_dose)
        axes[i, 1].set_title(f'Predicción @ z={z}')
        axes[i, 1].set_ylabel('Y')
        plt.colorbar(im1, ax=axes[i, 1])
        
        # GT
        im2 = axes[i, 2].imshow(target_vol[z, :, :], cmap='hot', vmin=0, vmax=max_dose)
        axes[i, 2].set_title(f'Ground Truth @ z={z}')
        axes[i, 2].set_ylabel('Y')
        plt.colorbar(im2, ax=axes[i, 2])
    
    plt.tight_layout()
    return fig

def plot_2d_slice_comparison(input_vol, pred_vol, target_vol, z_slice):
    """Genera comparación 2D de un slice específico: Input | Pred | GT | Errores"""
    max_dose = target_vol.max()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Slice 2D
    input_slice = input_vol[z_slice, :, :]
    pred_slice = pred_vol[z_slice, :, :]
    target_slice = target_vol[z_slice, :, :]
    
    # Row 1: Input, Pred, GT
    im0 = axes[0, 0].imshow(input_slice, cmap='hot')
    axes[0, 0].set_title(f'Input (z={z_slice})')
    plt.colorbar(im0, ax=axes[0, 0])
    
    im1 = axes[0, 1].imshow(pred_slice, cmap='hot')
    axes[0, 1].set_title(f'Predicción (z={z_slice})')
    plt.colorbar(im1, ax=axes[0, 1])
    
    im2 = axes[0, 2].imshow(target_slice, cmap='hot')
    axes[0, 2].set_title(f'Ground Truth (z={z_slice})')
    plt.colorbar(im2, ax=axes[0, 2])
    
    # Row 2: Input-GT, Pred-GT, Error%
    diff_input = np.abs(input_slice - target_slice)
    im3 = axes[1, 0].imshow(diff_input, cmap='viridis')
    axes[1, 0].set_title(f'Input - GT (z={z_slice})')
    plt.colorbar(im3, ax=axes[1, 0])
    
    diff_pred = np.abs(pred_slice - target_slice)
    im4 = axes[1, 1].imshow(diff_pred, cmap='viridis')
    axes[1, 1].set_title(f'Pred - GT (z={z_slice})')
    plt.colorbar(im4, ax=axes[1, 1])
    
    error_pct = np.zeros_like(target_slice, dtype=np.float32)
    mask = target_slice > 0.01 * max_dose
    error_pct[mask] = 100.0 * np.abs(pred_slice[mask] - target_slice[mask]) / (target_slice[mask] + 1e-8)
    im5 = axes[1, 2].imshow(error_pct, cmap='RdYlGn_r', vmin=0, vmax=30)
    axes[1, 2].set_title(f'Error % (z={z_slice})')
    plt.colorbar(im5, ax=axes[1, 2])
    
    plt.tight_layout()
    return fig


    """Helper para subplot individual"""
    max_val = volume.max()
    threshold = threshold_pct * max_val
    
    # Validar que threshold esté dentro del rango
    if max_val <= 0 or threshold > max_val:
        # Si threshold es inválido, usar 50% del máximo
        threshold = 0.5 * max_val
    
    if threshold <= 0:
        ax.text(0.5, 0.5, 0.5, 'Sin datos', transform=ax.transAxes, ha='center')
        ax.set_title(f"{title}\n(sin datos)")
        return
    
    try:
        verts, faces, _, _ = measure.marching_cubes(volume, level=threshold, spacing=(1, 1, 1))
    except ValueError:
        # Si marching_cubes falla, usar percentil 75
        threshold = np.percentile(volume[volume > 0], 75)
        verts, faces, _, _ = measure.marching_cubes(volume, level=threshold, spacing=(1, 1, 1))
    
    mesh = Poly3DCollection(verts[faces], alpha=0.3, edgecolor='none')
    mesh.set_facecolor(color)
    ax.add_collection3d(mesh)
    
    ax.set_xlim(0, volume.shape[0])
    ax.set_ylim(0, volume.shape[1])
    ax.set_zlim(0, volume.shape[2])
    
    ax.set_xlabel('Z')
    ax.set_ylabel('Y')
    ax.set_zlabel('X')
    ax.set_title(f"{title}\n(iso @ {threshold:.1f})")

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "=" * 70)
    print("VISUALIZACIÓN 3D: Input vs Predicción vs GT")
    print("=" * 70)
    
    # Cargar volúmenes
    input_vol, pred_vol, target_vol = load_volumes()
    print(f"\n✓ Volúmenes cargados: {input_vol.shape}")
    print(f"  Max dose - Input: {input_vol.max():.2f}")
    print(f"  Max dose - Pred: {pred_vol.max():.2f}")
    print(f"  Max dose - GT: {target_vol.max():.2f}")
    
    # Generar visualizaciones a diferentes thresholds
    for threshold in [0.3, 0.5, 0.7]:
        print(f"\n  Generando isosurfaces 3D completas @ {threshold*100}% max...")
        
        # Comparación completa (6 paneles)
        fig = plot_3d_comparison_complete(input_vol, pred_vol, target_vol, threshold_pct=threshold)
        output_path = OUTPUT_DIR / f"{PAIR_TO_VISUALIZE}_{INPUT_LEVEL}_3d_complete_{int(threshold*100)}pct.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    Guardado: {output_path}")
    
    # Visualización individual de predicción
    print(f"\n  Generando vista detallada predicción...")
    fig = plot_3d_isosurface(pred_vol, threshold_pct=0.5, title="Predicción 3D", color='blue', alpha=0.5)
    output_path = OUTPUT_DIR / f"{PAIR_TO_VISUALIZE}_{INPUT_LEVEL}_pred_3d_detail.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Guardado: {output_path}")
    
    # Cortes 2D en grilla para ver desfases dimensionales
    print(f"\n  Generando cortes 2D en grilla (6 slices)...")
    z_slices = [25, 50, 75, 100, 125, 150]
    fig = plot_multiple_2d_slices(input_vol, pred_vol, target_vol, z_slices)
    output_path = OUTPUT_DIR / f"{PAIR_TO_VISUALIZE}_{INPUT_LEVEL}_2d_slices_detail.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Guardado: {output_path}")
    
    # Cortes 2D en z específicos para ver errores locales
    print(f"\n  Generando cortes 2D de comparación...")
    for z_slice in [15, 75, 150]:
        if z_slice >= input_vol.shape[0]:
            continue
        fig = plot_2d_slice_comparison(input_vol, pred_vol, target_vol, z_slice)
        output_path = OUTPUT_DIR / f"{PAIR_TO_VISUALIZE}_{INPUT_LEVEL}_2d_slice_z{z_slice}.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    Guardado: {output_path}")
    
    print(f"\n{'='*70}")
    print(f"✓ Visualizaciones guardadas en: {OUTPUT_DIR}")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
