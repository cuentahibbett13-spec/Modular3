#!/usr/bin/env python3
"""
Análisis simple: Lee las métricas del denoising y las muestra.
"""

from pathlib import Path
import json
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Evalúa qué tan bien denoisea el modelo")
    parser.add_argument("--metrics", type=Path, default="results_inference/metrics.json", help="Archivo de métricas")
    args = parser.parse_args()
    
    if not args.metrics.exists():
        print(f"❌ {args.metrics} no encontrado")
        return
    
    # Leer métricas
    with open(args.metrics) as f:
        metrics = json.load(f)
    
    print("\n" + "="*60)
    print("📊 EVALUACIÓN DEL DENOISING")
    print("="*60)
    
    # Extraer arrays
    mse_vals = np.array([m['MSE'] for m in metrics])
    mae_vals = np.array([m['MAE'] for m in metrics])
    psnr_vals = np.array([m['PSNR'] for m in metrics])
    corr_vals = np.array([m['Corr'] for m in metrics])
    
    # Imprimir por sample
    print("\n📈 RESULTADOS POR SAMPLE:")
    print("-" * 60)
    print(f"{'Sample':<8} {'MSE':<12} {'MAE':<12} {'PSNR (dB)':<12} {'Correlación':<12}")
    print("-" * 60)
    for i, m in enumerate(metrics, 1):
        print(f"{i:<8} {m['MSE']:<12.6f} {m['MAE']:<12.6f} {m['PSNR']:<12.2f} {m['Corr']:<12.4f}")
    
    # Estadísticas generales
    print("\n" + "="*60)
    print("📊 ESTADÍSTICAS GENERALES:")
    print("="*60)
    
    print(f"\nMSE (Error Cuadrático Medio):")
    print(f"  Promedio: {mse_vals.mean():.6f}")
    print(f"  Desv.Est: {mse_vals.std():.6f}")
    print(f"  Min-Max:  {mse_vals.min():.6f} - {mse_vals.max():.6f}")
    print(f"  ➜ Qué significa: Menor es mejor. Mide diferencia cuadrática.")
    
    print(f"\nMAE (Error Medio Absoluto):")
    print(f"  Promedio: {mae_vals.mean():.6f}")
    print(f"  Desv.Est: {mae_vals.std():.6f}")
    print(f"  Min-Max:  {mae_vals.min():.6f} - {mae_vals.max():.6f}")
    print(f"  ➜ Qué significa: Error promedio en dosis. Menor = mejor predicción.")
    
    print(f"\nPSNR (Peak Signal-to-Noise Ratio):")
    print(f"  Promedio: {psnr_vals.mean():.2f} dB")
    print(f"  Desv.Est: {psnr_vals.std():.2f} dB")
    print(f"  Min-Max:  {psnr_vals.min():.2f} - {psnr_vals.max():.2f} dB")
    print(f"  ➜ Qué significa: Mayor es mejor. >20dB es bueno, >30dB es excelente.")
    
    print(f"\nCORRELACIÓN (Pearson):")
    print(f"  Promedio: {corr_vals.mean():.4f}")
    print(f"  Desv.Est: {corr_vals.std():.4f}")
    print(f"  Min-Max:  {corr_vals.min():.4f} - {corr_vals.max():.4f}")
    print(f"  ➜ Qué significa: 1.0 = predicción perfecta. >0.99 es muy bueno.")
    
    # Evaluación general
    print("\n" + "="*60)
    print("🎯 VEREDICTO:")
    print("="*60)
    
    if psnr_vals.mean() > 30 and corr_vals.mean() > 0.995:
        print("✅ EXCELENTE: El modelo denoisea muy bien los volúmenes de dosis")
    elif psnr_vals.mean() > 25 and corr_vals.mean() > 0.99:
        print("✅ BUENO: El modelo hace un buen trabajo en denoising")
    elif psnr_vals.mean() > 20 and corr_vals.mean() > 0.98:
        print("⚠️  ACEPTABLE: El modelo mejora la dosis pero hay margen")
    else:
        print("❌ POBRE: El modelo necesita mejoras")
    
    print("\n" + "="*60)
    print(f"📁 Visualizaciones en: results_inference/")
    print(f"   (Abre los PNG para ver input ruidoso vs predicción vs target)")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
