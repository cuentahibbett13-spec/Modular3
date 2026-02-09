#!/usr/bin/env python3
"""
Verificación: ¿Es el modelo solo un multiplicador trivial?
Compara la predicción IA vs escalado simple por factor
"""
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

def analyze_scaling_factor():
    """Verifica si el modelo es solo un multiplicador de eventos"""
    
    # Cargar métricas
    metrics_file = Path("runs/denoising_v2_residual/evaluation/metrics.json")
    if not metrics_file.exists():
        print("❌ Error: ejecuta evaluate_model.py primero")
        return
        
    # Cargar datos exportados
    exports_dir = Path("exports")
    if not exports_dir.exists():
        print("❌ Error: ejecuta export_predictions.py primero")
        return
    
    print("="*80)
    print("VERIFICACIÓN: ¿MODELO vs MULTIPLICADOR TRIVIAL?")
    print("="*80)
    print("Hipótesis nula: El modelo solo multiplica input × factor = pred")
    print("Si esto fuera cierto, sería inútil vs simular más eventos\n")
    
    # Encontrar casos con input 1M
    input_files = list(exports_dir.glob("*1M_input.npy"))
    if not input_files:
        input_files = list(exports_dir.glob("*_input.npy"))
        if not input_files:
            print("❌ No se encontraron archivos de input")
            return
    
    results = []
    
    for input_file in input_files:
        case_name = input_file.stem.replace("_input", "")
        
        # Cargar volúmenes
        input_vol = np.load(exports_dir / f"{case_name}_input.npy")
        pred_vol = np.load(exports_dir / f"{case_name}_pred.npy")  
        target_vol = np.load(exports_dir / f"{case_name}_target.npy")
        
        print(f"📊 Analizando: {case_name}")
        
        # 1. CALCULAR FACTOR DE ESCALA GLOBAL
        # Máscara para voxels significativos
        mask = target_vol > 0.01 * target_vol.max()
        
        if mask.sum() == 0:
            print(f"   ⚠️ Sin voxels significativos, saltando...")
            continue
            
        # Factor de escala promedio (target/input)
        input_masked = input_vol[mask]
        target_masked = target_vol[mask]
        pred_masked = pred_vol[mask]
        
        # Evitar división por cero
        valid_mask = input_masked > 1e-10
        input_valid = input_masked[valid_mask]
        target_valid = target_masked[valid_mask]
        pred_valid = pred_masked[valid_mask]
        
        if len(input_valid) == 0:
            print(f"   ⚠️ Sin datos válidos, saltando...")
            continue
        
        # Calcular factores de escala
        scale_factors = target_valid / input_valid
        mean_scale = np.mean(scale_factors)
        std_scale = np.std(scale_factors)
        
        print(f"   Factor de escala medio: {mean_scale:.2f}")
        print(f"   Desviación estándar: {std_scale:.2f}")
        
        # 2. CREAR PREDICCIÓN "INGENUA" (solo multiplicar)
        naive_pred = input_vol * mean_scale
        naive_masked = naive_pred[mask][valid_mask]
        
        # 3. COMPARAR PREDICCIONES
        mae_ai_vs_target = np.mean(np.abs(pred_valid - target_valid))
        mae_naive_vs_target = np.mean(np.abs(naive_valid - target_valid))
        mae_ai_vs_naive = np.mean(np.abs(pred_valid - naive_valid))
        
        # Correlaciones
        corr_ai_target = np.corrcoef(pred_valid, target_valid)[0, 1]
        corr_naive_target = np.corrcoef(naive_valid, target_valid)[0, 1] 
        corr_ai_naive = np.corrcoef(pred_valid, naive_valid)[0, 1]
        
        # 4. ANÁLISIS DE SIMILARIDAD
        # Si el modelo fuera solo un multiplicador: corr_ai_naive ≈ 1.0
        is_trivial_threshold = 0.98  # Muy alta correlación = modelo trivial
        
        print(f"   📊 COMPARACIÓN DE PREDICCIONES:")
        print(f"      IA vs Target:     MAE={mae_ai_vs_target:.3f}, r={corr_ai_target:.4f}")
        print(f"      Naïve vs Target:  MAE={mae_naive_vs_target:.3f}, r={corr_naive_target:.4f}")
        print(f"      IA vs Naïve:      MAE={mae_ai_vs_naive:.3f}, r={corr_ai_naive:.4f}")
        
        # Factores de mejora
        improvement_vs_naive = mae_naive_vs_target / mae_ai_vs_target if mae_ai_vs_target > 0 else 1
        
        print(f"   🎯 MEJORA IA vs MULTIPLICADOR SIMPLE:")
        print(f"      Factor de mejora MAE: {improvement_vs_naive:.2f}x")
        
        # Veredicto
        if corr_ai_naive > is_trivial_threshold:
            verdict = "⚠️ POSIBLE MULTIPLICADOR TRIVIAL"
            color = "red"
        elif improvement_vs_naive > 2.0:
            verdict = "✅ MODELO INTELIGENTE (>2x mejor)"
            color = "green"
        elif improvement_vs_naive > 1.5:
            verdict = "🔶 MODELO MODERADAMENTE ÚTIL"
            color = "orange"
        else:
            verdict = "❌ MODELO POCO ÚTIL"
            color = "red"
        
        print(f"   {verdict}")
        
        # 5. ANÁLISIS ESPACIAL
        analyze_spatial_patterns(input_vol, pred_vol, naive_pred, target_vol, case_name, mask)
        
        results.append({
            'case': case_name,
            'mean_scale': float(mean_scale),
            'std_scale': float(std_scale),
            'mae_ai_vs_target': float(mae_ai_vs_target),
            'mae_naive_vs_target': float(mae_naive_vs_target),
            'mae_ai_vs_naive': float(mae_ai_vs_naive),
            'corr_ai_target': float(corr_ai_target),
            'corr_naive_target': float(corr_naive_target),
            'corr_ai_naive': float(corr_ai_naive),
            'improvement_factor': float(improvement_vs_naive),
            'verdict': verdict
        })
        
        print()
    
    # 6. RESUMEN GLOBAL
    print("="*80)
    print("📊 RESUMEN GLOBAL")
    print("="*80)
    
    if not results:
        print("❌ No se pudo analizar ningún caso")
        return
    
    avg_corr_ai_naive = np.mean([r['corr_ai_naive'] for r in results])
    avg_improvement = np.mean([r['improvement_factor'] for r in results])
    
    print(f"Correlación promedio IA vs Multiplicador: {avg_corr_ai_naive:.4f}")
    print(f"Factor de mejora promedio: {avg_improvement:.2f}x")
    
    # Veredicto final
    trivial_cases = sum(1 for r in results if 'TRIVIAL' in r['verdict'])
    intelligent_cases = sum(1 for r in results if 'INTELIGENTE' in r['verdict'])
    
    print(f"\n🎯 VEREDICTO FINAL:")
    print(f"   Casos triviales: {trivial_cases}/{len(results)}")
    print(f"   Casos inteligentes: {intelligent_cases}/{len(results)}")
    
    if avg_corr_ai_naive > 0.95:
        final_verdict = "⚠️ EL MODELO ES MAYORMENTE UN MULTIPLICADOR TRIVIAL"
    elif avg_improvement > 2.0:
        final_verdict = "✅ EL MODELO ES INTELIGENTE Y ÚTIL"
    else:
        final_verdict = "🔶 EL MODELO TIENE UTILIDAD MODERADA"
    
    print(f"\n{final_verdict}")
    
    if avg_corr_ai_naive > 0.95:
        print("\n💡 INTERPRETACIÓN:")
        print("   El modelo aprendió principalmente a escalar por un factor.")
        print("   Esto no es mejor que simular más eventos directamente.")
        print("   Considera revisar la arquitectura o función de pérdida.")
    else:
        print("\n💡 INTERPRETACIÓN:")
        print("   El modelo aprendió patrones espaciales complejos.")
        print("   Va más allá del simple escalado de eventos.")
        print("   Es genuinamente útil para denoising de dosis.")

def analyze_spatial_patterns(input_vol, pred_vol, naive_pred, target_vol, case_name, mask):
    """Analiza patrones espaciales para verificar inteligencia del modelo"""
    
    plots_dir = Path("scaling_analysis_plots")
    plots_dir.mkdir(exist_ok=True)
    
    # Seleccionar slice con más información
    slice_doses = [target_vol[z].max() for z in range(target_vol.shape[0])]
    best_z = np.argmax(slice_doses)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    vmax = target_vol.max()
    
    # Fila 1: Volúmenes
    im0 = axes[0,0].imshow(input_vol[best_z], cmap='hot', vmin=0, vmax=vmax, aspect='auto')
    axes[0,0].set_title('Input (ruidoso)')
    axes[0,0].axis('off')
    plt.colorbar(im0, ax=axes[0,0], fraction=0.046)
    
    im1 = axes[0,1].imshow(pred_vol[best_z], cmap='hot', vmin=0, vmax=vmax, aspect='auto')
    axes[0,1].set_title('Predicción IA')
    axes[0,1].axis('off')
    plt.colorbar(im1, ax=axes[0,1], fraction=0.046)
    
    im2 = axes[0,2].imshow(naive_pred[best_z], cmap='hot', vmin=0, vmax=vmax, aspect='auto')
    axes[0,2].set_title('Multiplicador Simple')
    axes[0,2].axis('off')
    plt.colorbar(im2, ax=axes[0,2], fraction=0.046)
    
    # Fila 2: Diferencias
    diff_ai = np.abs(pred_vol[best_z] - target_vol[best_z])
    diff_naive = np.abs(naive_pred[best_z] - target_vol[best_z])
    diff_ai_naive = np.abs(pred_vol[best_z] - naive_pred[best_z])
    
    diff_max = max(diff_ai.max(), diff_naive.max())
    
    im3 = axes[1,0].imshow(diff_ai, cmap='viridis', vmin=0, vmax=diff_max, aspect='auto')
    axes[1,0].set_title('|IA - Target|')
    axes[1,0].axis('off')
    plt.colorbar(im3, ax=axes[1,0], fraction=0.046)
    
    im4 = axes[1,1].imshow(diff_naive, cmap='viridis', vmin=0, vmax=diff_max, aspect='auto')
    axes[1,1].set_title('|Multiplicador - Target|')
    axes[1,1].axis('off')
    plt.colorbar(im4, ax=axes[1,1], fraction=0.046)
    
    im5 = axes[1,2].imshow(diff_ai_naive, cmap='RdBu_r', aspect='auto')
    axes[1,2].set_title('IA vs Multiplicador')
    axes[1,2].axis('off')
    plt.colorbar(im5, ax=axes[1,2], fraction=0.046)
    
    fig.suptitle(f'Análisis Espacial - {case_name} (z={best_z})', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plots_dir / f'{case_name}_spatial_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Gráfica espacial guardada: {case_name}_spatial_analysis.png")

if __name__ == "__main__":
    analyze_scaling_factor()