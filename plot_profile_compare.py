#!/usr/bin/env python3
"""
Plot dose profile comparison:
- Target (29.4M)
- Input scaled by factor (e.g., 29.4x)
- Prediction (U-Net)

Outputs a PNG with PDD and lateral profile.
"""
import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


# ----------------------------
# Model definition (same as training)
# ----------------------------
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


def load_volume(mhd_path):
    return sitk.GetArrayFromImage(sitk.ReadImage(str(mhd_path))).astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot PDD and lateral profile")
    parser.add_argument("--pair", default="pair_021", help="pair directory name")
    parser.add_argument("--level", default="input_1M", help="input level")
    parser.add_argument("--scale", type=float, default=29.4, help="scale factor for input")
    parser.add_argument("--dataset-root", default="dataset_pilot", help="dataset root")
    parser.add_argument("--model", default="runs/denoising_v2_residual/best_model.pt", help="model path")
    parser.add_argument("--out", default="runs/denoising_v2_residual/evaluation/profile_compare.png", help="output png")
    parser.add_argument("--skip-pred", action="store_true", help="skip model prediction")
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--overlap", type=int, default=16)
    parser.add_argument("--device", default=None, help="cuda or cpu")
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    val_dir = dataset_root / "val"
    pair_dir = val_dir / args.pair

    if not pair_dir.exists():
        raise FileNotFoundError(f"Pair not found: {pair_dir}")

    # Resolve target index
    n_targets = len(list(dataset_root.glob("target_*")))
    pair_num = int(args.pair.split("_")[-1])
    target_idx = ((pair_num - 1) % n_targets) + 1
    target_mhd = dataset_root / f"target_{target_idx}" / "dose_edep.mhd"

    if not target_mhd.exists():
        raise FileNotFoundError(f"Target not found: {target_mhd}")

    # Input path
    input_mhd = pair_dir / f"{args.level}.mhd"
    if not input_mhd.exists():
        input_mhd = pair_dir / args.level / "dose_edep.mhd"

    if not input_mhd.exists():
        raise FileNotFoundError(f"Input not found: {input_mhd}")

    target = load_volume(target_mhd)
    input_vol = load_volume(input_mhd)
    max_dose = float(target.max())

    # Scaled input
    scaled_input = input_vol * args.scale

    pred_vol = None
    if not args.skip_pred:
        if torch is None:
            raise RuntimeError("torch is not available. Use --skip-pred.")
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(device)

        model = ResidualUNet3D(base_channels=16).to(device)
        checkpoint = torch.load(args.model, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        input_norm = input_vol / (max_dose + 1e-8)
        pred_norm = sliding_window_inference(
            model,
            input_norm,
            device,
            patch_size=args.patch_size,
            overlap=args.overlap,
        )
        pred_vol = pred_norm * max_dose

    # PDD (max per z)
    pdd_target = np.array([target[z].max() for z in range(target.shape[0])])
    pdd_scaled = np.array([scaled_input[z].max() for z in range(scaled_input.shape[0])])
    pdd_pred = np.array([pred_vol[z].max() for z in range(pred_vol.shape[0])]) if pred_vol is not None else None

    # Lateral profile at z of max dose
    z_max = int(np.argmax(pdd_target))
    y_center = target.shape[1] // 2

    prof_target = target[z_max, y_center, :]
    prof_scaled = scaled_input[z_max, y_center, :]
    prof_pred = pred_vol[z_max, y_center, :] if pred_vol is not None else None

    # Normalize to target max
    denom = max_dose if max_dose > 0 else 1.0
    pdd_target_n = pdd_target / denom * 100.0
    pdd_scaled_n = pdd_scaled / denom * 100.0
    pdd_pred_n = pdd_pred / denom * 100.0 if pdd_pred is not None else None

    prof_target_n = prof_target / denom * 100.0
    prof_scaled_n = prof_scaled / denom * 100.0
    prof_pred_n = prof_pred / denom * 100.0 if prof_pred is not None else None

    # Plot
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # PDD
    axes[0].plot(pdd_target_n, "k-", linewidth=2, label="Target (29.4M)")
    axes[0].plot(pdd_scaled_n, "r--", linewidth=1.5, label=f"Input x{args.scale:g}")
    if pdd_pred_n is not None:
        axes[0].plot(pdd_pred_n, "b-", linewidth=1.8, label="Pred (U-Net)")
    axes[0].set_title("PDD (Percent Depth Dose)")
    axes[0].set_xlabel("Z layer")
    axes[0].set_ylabel("Dose (% of target max)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Lateral profile
    x_axis = np.arange(len(prof_target_n))
    axes[1].plot(x_axis, prof_target_n, "k-", linewidth=2, label="Target (29.4M)")
    axes[1].plot(x_axis, prof_scaled_n, "r--", linewidth=1.5, label=f"Input x{args.scale:g}")
    if prof_pred_n is not None:
        axes[1].plot(x_axis, prof_pred_n, "b-", linewidth=1.8, label="Pred (U-Net)")
    axes[1].set_title(f"Lateral profile (z={z_max}, y={y_center})")
    axes[1].set_xlabel("X index")
    axes[1].set_ylabel("Dose (% of target max)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved: {out_path}")
    print(f"Target: {target_mhd}")
    print(f"Input:  {input_mhd}")
    if pred_vol is not None:
        print(f"Model:  {args.model}")


if __name__ == "__main__":
    main()
