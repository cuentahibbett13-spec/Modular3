#!/usr/bin/env python3
"""
Script de análisis de eficiencia: Estadísticas del denoising, velocidad, memoria.
"""

from pathlib import Path
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo para cluster
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json
import time

from training.model import UNet3D


def analyze_inference_speed(model, val_ds, device, num_samples=5):
    """Mide velocidad de inferencia."""
    print("\n⏱️  ANÁLISIS DE VELOCIDAD")
    print("=" * 50)
    
    times = []
    model.eval()
    
    with torch.no_grad():
        for sample_idx in range(min(num_samples, len(val_ds))):
            batch = val_ds[sample_idx]
            inp = batch["input"].unsqueeze(0).to(device)
            
            # Warm-up
            _ = model(inp)
            
            # Medición
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t0 = time.time()
            pred = model(inp)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t1 = time.time()
            
            elapsed = (t1 - t0) * 1000  # ms
            times.append(elapsed)
            print(f"Sample {sample_idx + 1}: {elapsed:.2f} ms")
            
            # Liberar memoria
            del inp, pred, batch
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    times = np.array(times)
    print(f"\nPromedio: {times.mean():.2f} ± {times.std():.2f} ms")
    print(f"Min: {times.min():.2f} ms | Max: {times.max():.2f} ms")
    
    return times


def analyze_memory_usage(model, device):
    """Analiza uso de memoria."""
    print("\n💾 ANÁLISIS DE MEMORIA")
    print("=" * 50)
    
    # Parámetros del modelo
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
    
    print(f"Total de parámetros: {total_params:,.0f}")
    print(f"Parámetros entrenables: {trainable_params:,.0f}")
    print(f"Tamaño del modelo: {model_size:.2f} MB")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} GB")
    
    return {"total_params": total_params,
            "model_size": model_size}


def plot_metrics_summary(metrics_file: Path, output_dir: Path):
    """Grafica resumen de métricas desde el archivo JSON."""
    if not metrics_file.exists():
        print(f"⚠️  {metrics_file} no encontrado, saltando gráficos")
        return
    
    try:
        with open(metrics_file) as f:
            metrics_list = json.load(f)
        
        metrics_array = {k: [m[k] for m in metrics_list] for k in metrics_list[0].keys()}
        
        fig = plt.figure(figsize=(14, 8))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # MSE
        ax = fig.add_subplot(gs[0, 0])
        ax.bar(range(len(metrics_array['MSE'])), metrics_array['MSE'], color='steelblue', alpha=0.7)
        ax.set_xlabel('Sample')
        ax.set_ylabel('MSE')
        ax.set_title('Mean Squared Error por Sample')
        ax.grid(axis='y', alpha=0.3)
        
        # MAE
        ax = fig.add_subplot(gs[0, 1])
        ax.bar(range(len(metrics_array['MAE'])), metrics_array['MAE'], color='coral', alpha=0.7)
        ax.set_xlabel('Sample')
        ax.set_ylabel('MAE')
        ax.set_title('Mean Absolute Error por Sample')
        ax.grid(axis='y', alpha=0.3)
        
        # PSNR
        ax = fig.add_subplot(gs[1, 0])
        ax.bar(range(len(metrics_array['PSNR'])), metrics_array['PSNR'], color='seagreen', alpha=0.7)
        ax.set_xlabel('Sample')
        ax.set_ylabel('PSNR (dB)')
        ax.set_title('Peak Signal-to-Noise Ratio por Sample')
        ax.grid(axis='y', alpha=0.3)
        
        # Correlación
        ax = fig.add_subplot(gs[1, 1])
        ax.bar(range(len(metrics_array['Corr'])), metrics_array['Corr'], color='mediumpurple', alpha=0.7)
        ax.set_xlabel('Sample')
        ax.set_ylabel('Correlación')
        ax.set_title('Correlación de Pearson por Sample')
        ax.set_ylim([0.9, 1.0])
        ax.grid(axis='y', alpha=0.3)
        
        fig.suptitle('Resumen de Métricas de Denoising', fontsize=14, fontweight='bold')
        plt.savefig(output_dir / 'metrics_summary.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        # Estadísticas
        print("\n📊 ESTADÍSTICAS DE CALIDAD")
        print("=" * 50)
        for metric_name, values in metrics_array.items():
            print(f"{metric_name:8s}: {np.mean(values):8.4f} ± {np.std(values):.4f}")
        
    except Exception as e:
        print(f"⚠️  Error al procesar métricas: {e}")


def main():
    parser = argparse.ArgumentParser(description="Análisis de eficiencia de inferencia")
    parser.add_argument("--data-root", type=Path, default="dataset_pilot", help="Dataset root")
    parser.add_argument("--checkpoint", type=Path, default="runs/denoising/best.pt", help="Model checkpoint")
    parser.add_argument("--output-dir", type=Path, default="results_analysis", help="Output directory")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to analyze")
    parser.add_argument("--device", default="auto", help="Device: cuda/cpu/auto")
    args = parser.parse_args()
    
    # Setup
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"🔧 Device: {device}")
    
    # Load model
    print(f"📦 Cargando modelo desde {args.checkpoint}")
    model = UNet3D(in_ch=1, out_ch=1, base_ch=32).to(device)
    ckpt = torch.load(str(args.checkpoint), map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print("✅ Modelo cargado")
    
    # Load dataset (con manejo de errores)
    try:
        from training.dataset import DosePairDataset
        val_ds = DosePairDataset(
            root_dir=args.data_root,
            split="val",
            patch_size=(64, 64, 64),
            cache_size=0,
            normalize=True,
            seed=4321,
        )
        print(f"✅ Dataset: {len(val_ds)} samples")
        
        # Análisis
        analyze_memory_usage(model, device)
        times = analyze_inference_speed(model, val_ds, device, args.num_samples)
        
    except Exception as e:
        print(f"⚠️  Error con dataset: {e}")
        # Continuar solo con análisis de modelo
        analyze_memory_usage(model, device)
    
    # Cargar y graficar métricas si existen
    metrics_file = Path("results_inference/metrics.json")
    if metrics_file.exists():
        plot_metrics_summary(metrics_file, args.output_dir)
    else:
        print(f"⚠️  {metrics_file} no encontrado")
    
    print("\n✅ Análisis completado")
    print(f"📁 Resultados guardados en: {args.output_dir}")


if __name__ == "__main__":
    main()

