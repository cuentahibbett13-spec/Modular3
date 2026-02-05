#!/usr/bin/env python3
"""
Script de inferencia mejorado con mejor manejo de memoria y errores.
"""

from pathlib import Path
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json
import gc

from training.dataset import DosePairDataset
from training.model import UNet3D


def load_model(checkpoint_path: Path, device: torch.device) -> UNet3D:
    """Carga el modelo desde checkpoint."""
    model = UNet3D(in_ch=1, out_ch=1, base_ch=32).to(device)
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def infer_batch(model, inp: torch.Tensor, device: torch.device) -> np.ndarray:
    """Realiza inferencia en un batch."""
    with torch.no_grad():
        inp = inp.to(device)
        pred = model(inp)
        pred = pred.cpu().squeeze().numpy()
    return pred


def plot_slices(input_vol, pred_vol, target_vol, output_path: Path, title: str):
    """Grafica 3 slices axiales: input, predicción, target."""
    try:
        z_idx = input_vol.shape[0] // 2
        
        fig = plt.figure(figsize=(15, 5))
        gs = GridSpec(1, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        slices = [
            (input_vol[z_idx], "Input (Ruidoso)"),
            (pred_vol[z_idx], "Predicción"),
            (target_vol[z_idx], "Ground Truth"),
        ]
        
        vmin = min(v[v > 0].min() for v, _ in slices if v[v > 0].size > 0)
        vmax = max(v.max() for v, _ in slices)
        
        for idx, (data, label) in enumerate(slices):
            ax = fig.add_subplot(gs[0, idx])
            im = ax.imshow(data, cmap="hot", vmin=vmin, vmax=vmax)
            ax.set_title(label, fontsize=12, fontweight="bold")
            ax.axis("off")
            plt.colorbar(im, ax=ax, label="Dosis (a.u.)")
        
        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(str(output_path), dpi=100, bbox_inches="tight")
        plt.close(fig)
        
        # Limpiar memoria
        del fig, gs, slices
        
    except Exception as e:
        print(f"   ⚠️  Error al crear visualización: {e}")


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """Calcula métricas de calidad."""
    mse = np.mean((pred - target) ** 2)
    mae = np.mean(np.abs(pred - target))
    
    max_val = target.max()
    psnr = 20 * np.log10(max_val / np.sqrt(mse)) if mse > 0 else float('inf')
    
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    corr = np.corrcoef(pred_flat, target_flat)[0, 1]
    
    return {
        'MSE': float(mse),
        'MAE': float(mae),
        'PSNR': float(psnr),
        'Corr': float(corr),
    }


def main():
    parser = argparse.ArgumentParser(description="Inferencia con modelo de denoising")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results_inference"))
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    print(f"🔧 Device: {device}")
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Cargar modelo
        print(f"📦 Cargando modelo...")
        model = load_model(args.checkpoint, device)
        print(f"✅ Modelo cargado desde {args.checkpoint}")
        
        # Dataset de validación
        print(f"📂 Cargando dataset...")
        val_ds = DosePairDataset(
            root_dir=args.data_root,
            split="val",
            patch_size=(64, 64, 64),
            cache_size=0,
            normalize=True,
            seed=4321,
        )
        print(f"✅ Dataset: {len(val_ds)} samples")
        
        # Inferencia
        metrics_list = []
        num_to_process = min(args.num_samples, len(val_ds))
        
        for sample_idx in range(num_to_process):
            print(f"\n📊 Procesando sample {sample_idx + 1}/{num_to_process}")
            
            try:
                # Cargar batch
                batch = val_ds[sample_idx]
                inp = batch["input"].unsqueeze(0)
                tgt = batch["target"].squeeze(0).numpy()
                
                # Inferencia
                print(f"   🔄 Ejecutando inferencia...")
                pred = infer_batch(model, inp, device)
                
                # Métricas
                print(f"   📏 Calculando métricas...")
                metrics = compute_metrics(pred, tgt)
                metrics_list.append(metrics)
                print(f"   MSE: {metrics['MSE']:.6f} | MAE: {metrics['MAE']:.6f}")
                print(f"   PSNR: {metrics['PSNR']:.2f} dB | Corr: {metrics['Corr']:.4f}")
                
                # Visualizar
                print(f"   🖼️  Guardando visualización...")
                title = f"Sample {sample_idx + 1} | PSNR={metrics['PSNR']:.2f}dB | Corr={metrics['Corr']:.4f}"
                out_path = args.output_dir / f"sample_{sample_idx + 1:02d}.png"
                inp_vis = inp.squeeze().cpu().numpy()
                plot_slices(inp_vis, pred, tgt, out_path, title)
                print(f"   ✅ Guardado: {out_path.name}")
                
                # Limpiar memoria
                del batch, inp, tgt, pred, inp_vis, metrics
                gc.collect()
                
            except Exception as e:
                print(f"   ❌ Error procesando sample {sample_idx + 1}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Resumen
        if metrics_list:
            print("\n" + "="*50)
            print("📈 RESUMEN DE MÉTRICAS")
            print("="*50)
            avg_mse = np.mean([m["MSE"] for m in metrics_list])
            avg_mae = np.mean([m["MAE"] for m in metrics_list])
            avg_psnr = np.mean([m["PSNR"] for m in metrics_list])
            avg_corr = np.mean([m["Corr"] for m in metrics_list])
            
            print(f"MSE promedio:  {avg_mse:.6f}")
            print(f"MAE promedio:  {avg_mae:.6f}")
            print(f"PSNR promedio: {avg_psnr:.2f} dB")
            print(f"Corr promedio: {avg_corr:.4f}")
            
            # Guardar métricas en JSON
            metrics_file = args.output_dir / "metrics.json"
            with open(metrics_file, "w") as f:
                json.dump(metrics_list, f, indent=2)
            print(f"✅ Métricas guardadas en {metrics_file}")
            print(f"✅ Resultados guardados en {args.output_dir}")
        else:
            print("\n❌ No se procesaron samples exitosamente")
            
    except Exception as e:
        print(f"\n❌ Error fatal: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
