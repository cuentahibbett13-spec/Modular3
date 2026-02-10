#!/usr/bin/env python3
"""
Test model on unseen event levels (e.g., 100k, 500k, 3M, etc.)
not present in training set (1M, 2M, 5M, 10M).

Usage:
  python3 test_unseen_levels.py --input path/to/dose_100k.mhd --target path/to/target/dose_edep.mhd
"""
import os
os.environ["MIOPEN_DEBUG_DISABLE_FIND_DB"] = "1"

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================================
# MODEL
# ============================================================================
class ResidualUNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=16):
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
            nn.ReLU(inplace=True),
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
# INFERENCE
# ============================================================================
def sliding_window_inference(model, volume, device, patch_size=96, overlap=16):
    model.eval()
    z, y, x = volume.shape
    step = patch_size - overlap

    pad_z = (step - (z % step)) % step
    pad_y = (step - (y % step)) % step
    pad_x = (step - (x % step)) % step

    padded = np.pad(volume, ((0, pad_z), (0, pad_y), (0, pad_x)), mode="constant")
    pz, py, px = padded.shape

    output = np.zeros_like(padded)
    weight_map = np.zeros_like(padded)

    w = np.ones((patch_size, patch_size, patch_size), dtype=np.float32)
    margin = overlap // 2
    if margin > 0:
        for i in range(margin):
            fade = (i + 1) / margin
            w[i, :, :] *= fade
            w[-(i + 1), :, :] *= fade
            w[:, i, :] *= fade
            w[:, -(i + 1), :] *= fade
            w[:, :, i] *= fade
            w[:, :, -(i + 1)] *= fade

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
            patch = padded[zs:zs + patch_size, ys:ys + patch_size, xs:xs + patch_size]
            patch_t = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0).to(device)
            pred = model(patch_t)
            pred_np = pred.squeeze().cpu().numpy()
            output[zs:zs + patch_size, ys:ys + patch_size, xs:xs + patch_size] += pred_np * w
            weight_map[zs:zs + patch_size, ys:ys + patch_size, xs:xs + patch_size] += w

    weight_map = np.maximum(weight_map, 1e-8)
    output = output / weight_map
    return output[:z, :y, :x]


# ============================================================================
# METRICS
# ============================================================================
def calc_psnr(pred, target, max_val):
    mse = np.mean((pred - target) ** 2)
    if mse < 1e-15:
        return 100.0
    return 10 * np.log10(max_val ** 2 / mse)


def calc_metrics(input_vol, pred_vol, target_vol, max_dose):
    """Calculate comprehensive metrics"""
    metrics = {}
    
    # Global PSNR
    psnr_input = calc_psnr(input_vol, target_vol, max_dose)
    psnr_pred = calc_psnr(pred_vol, target_vol, max_dose)
    metrics['psnr_input'] = psnr_input
    metrics['psnr_pred'] = psnr_pred
    metrics['psnr_gain'] = psnr_pred - psnr_input
    
    # MAE
    mae_input = np.mean(np.abs(input_vol - target_vol))
    mae_pred = np.mean(np.abs(pred_vol - target_vol))
    metrics['mae_input'] = mae_input
    metrics['mae_pred'] = mae_pred
    metrics['mae_reduction_%'] = (mae_input - mae_pred) / mae_input * 100
    
    # High dose region (>10% max)
    mask_high = target_vol > 0.10 * max_dose
    if mask_high.sum() > 0:
        mae_high_input = np.mean(np.abs(input_vol[mask_high] - target_vol[mask_high]))
        mae_high_pred = np.mean(np.abs(pred_vol[mask_high] - target_vol[mask_high]))
        rel_high_input = mae_high_input / max_dose * 100
        rel_high_pred = mae_high_pred / max_dose * 100
        
        metrics['high_dose_mae_input'] = mae_high_input
        metrics['high_dose_mae_pred'] = mae_high_pred
        metrics['high_dose_rel_input_%'] = rel_high_input
        metrics['high_dose_rel_pred_%'] = rel_high_pred
    
    # PDD correlation
    pdd_input = np.array([input_vol[z].max() for z in range(input_vol.shape[0])])
    pdd_pred = np.array([pred_vol[z].max() for z in range(pred_vol.shape[0])])
    pdd_target = np.array([target_vol[z].max() for z in range(target_vol.shape[0])])
    
    corr_input = np.corrcoef(pdd_input, pdd_target)[0, 1]
    corr_pred = np.corrcoef(pdd_pred, pdd_target)[0, 1]
    
    metrics['pdd_corr_input'] = corr_input
    metrics['pdd_corr_pred'] = corr_pred
    
    return metrics


# ============================================================================
# VISUALIZATION
# ============================================================================
def plot_results(input_vol, pred_vol, target_vol, metrics, output_path, label):
    """Create comparison plots"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Find slice with max dose
    pdd_target = np.array([target_vol[z].max() for z in range(target_vol.shape[0])])
    z_max = int(np.argmax(pdd_target))
    
    vmax = target_vol[z_max].max()
    
    # Row 1: Dose distributions
    im0 = axes[0, 0].imshow(input_vol[z_max], cmap='hot', vmin=0, vmax=vmax, aspect='auto')
    axes[0, 0].set_title(f'Input ({label})', fontsize=12)
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)
    
    im1 = axes[0, 1].imshow(pred_vol[z_max], cmap='hot', vmin=0, vmax=vmax, aspect='auto')
    axes[0, 1].set_title('Prediction (U-Net)', fontsize=12)
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    im2 = axes[0, 2].imshow(target_vol[z_max], cmap='hot', vmin=0, vmax=vmax, aspect='auto')
    axes[0, 2].set_title('Target (29.4M)', fontsize=12)
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
    
    # Row 2: Errors
    diff_input = np.abs(input_vol[z_max] - target_vol[z_max])
    diff_pred = np.abs(pred_vol[z_max] - target_vol[z_max])
    diff_max = max(diff_input.max(), diff_pred.max())
    
    im3 = axes[1, 0].imshow(diff_input, cmap='viridis', vmin=0, vmax=diff_max, aspect='auto')
    axes[1, 0].set_title('|Input - Target|', fontsize=12)
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
    
    im4 = axes[1, 1].imshow(diff_pred, cmap='viridis', vmin=0, vmax=diff_max, aspect='auto')
    axes[1, 1].set_title('|Pred - Target|', fontsize=12)
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)
    
    # PDD comparison
    pdd_input = np.array([input_vol[z].max() for z in range(input_vol.shape[0])])
    pdd_pred = np.array([pred_vol[z].max() for z in range(pred_vol.shape[0])])
    pdd_target = np.array([target_vol[z].max() for z in range(target_vol.shape[0])])
    
    axes[1, 2].plot(pdd_target, 'k-', linewidth=2, label='Target', alpha=0.8)
    axes[1, 2].plot(pdd_input, 'r--', linewidth=1, label=f'Input ({label})', alpha=0.6)
    axes[1, 2].plot(pdd_pred, 'b-', linewidth=1.5, label='Pred', alpha=0.8)
    axes[1, 2].set_xlabel('Z layer')
    axes[1, 2].set_ylabel('Max dose')
    axes[1, 2].set_title('PDD Comparison')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    for ax in axes.flat[:6]:
        ax.axis('off') if ax != axes[1, 2] else None
    
    fig.suptitle(f'Unseen Level Test: {label} | z={z_max}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Test model on unseen event levels")
    parser.add_argument("--input", required=True, help="Path to input .mhd file")
    parser.add_argument("--target", required=True, help="Path to target .mhd file")
    parser.add_argument("--label", default="100k", help="Label for this input (e.g., '100k', '500k')")
    parser.add_argument("--model", default="runs/denoising_v2_residual/best_model.pt", help="Model checkpoint")
    parser.add_argument("--output-dir", default="runs/denoising_v2_residual/unseen_levels", help="Output directory")
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--overlap", type=int, default=16)
    parser.add_argument("--device", default=None, help="cuda or cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n" + "=" * 70)
    print(f"TEST UNSEEN LEVEL: {args.label}")
    print("=" * 70)
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    
    # Load volumes
    print(f"\n[1/4] Loading volumes...")
    input_vol = sitk.GetArrayFromImage(sitk.ReadImage(args.input)).astype(np.float32)
    target_vol = sitk.GetArrayFromImage(sitk.ReadImage(args.target)).astype(np.float32)
    max_dose = target_vol.max()
    
    print(f"  Input:  {input_vol.shape}, max={input_vol.max():.4f}")
    print(f"  Target: {target_vol.shape}, max={max_dose:.4f}")
    
    # Load model
    print(f"\n[2/4] Loading model...")
    model = ResidualUNet3D(base_channels=16).to(device)
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print(f"  Model: {args.model}")
    print(f"  Device: {device}")
    
    # Inference
    print(f"\n[3/4] Running inference...")
    input_norm = input_vol / (max_dose + 1e-8)
    pred_norm = sliding_window_inference(model, input_norm, device, args.patch_size, args.overlap)
    pred_vol = pred_norm * max_dose
    print(f"  Prediction: {pred_vol.shape}, max={pred_vol.max():.4f}")
    
    # Metrics
    print(f"\n[4/4] Computing metrics...")
    metrics = calc_metrics(input_vol, pred_vol, target_vol, max_dose)
    
    print("\n" + "-" * 70)
    print("RESULTS:")
    print("-" * 70)
    print(f"  PSNR:  {metrics['psnr_input']:.1f} → {metrics['psnr_pred']:.1f} dB (gain: +{metrics['psnr_gain']:.1f} dB)")
    print(f"  MAE:   {metrics['mae_input']:.4f} → {metrics['mae_pred']:.4f} (reduction: {metrics['mae_reduction_%']:.1f}%)")
    
    if 'high_dose_rel_pred_%' in metrics:
        print(f"  High dose error: {metrics['high_dose_rel_input_%']:.2f}% → {metrics['high_dose_rel_pred_%']:.2f}%")
    
    print(f"  PDD correlation: {metrics['pdd_corr_input']:.4f} → {metrics['pdd_corr_pred']:.4f}")
    
    # Save
    plot_path = output_dir / f"test_{args.label}.png"
    plot_results(input_vol, pred_vol, target_vol, metrics, plot_path, args.label)
    print(f"\n  Saved: {plot_path}")
    
    # Save metrics
    import json
    metrics_path = output_dir / f"metrics_{args.label}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: {metrics_path}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
