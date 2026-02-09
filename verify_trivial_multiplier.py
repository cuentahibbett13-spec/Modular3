#!/usr/bin/env python3
"""
Verificación: ¿Es el modelo solo un multiplicador trivial?
Análisis independiente para cluster - genera datos sintéticos si no hay modelo
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Para cluster sin display
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path
import argparse

def create_synthetic_data():
    """Crea datos sintéticos para demostrar el análisis"""
    print("📦 Generando datos sintéticos para demonstración...")
    
    # Simulación de volumen de dosis 3D realista
    shape = (64, 96, 96)  # z, y, x
    
    # Target: distribución de dosis con gradiente radial
    z, y, x = np.meshgrid(np.linspace(-1, 1, shape[0]),
                          np.linspace(-1, 1, shape[1]), 
                          np.linspace(-1, 1, shape[2]), indexing='ij')
    
    # Centro del tumor
    center_dose = np.exp(-(z**2 + y**2 + x**2) / 0.4)
    
    # Agregar estructura anatómica
    organ_dose = 0.3 * np.exp(-((z-0.5)**2 + y**2 + x**2) / 0.2)
    target_vol = 100 * (center_dose + organ_dose)
    
    # Input: versión con menos eventos (ruidosa)
    events_ratio = 1/29  # De 1M a 29M eventos
    input_vol = target_vol * events_ratio
    
    # Agregar ruido Poisson realista
    noise_scale = 0.1
    input_vol += np.random.poisson(input_vol * noise_scale) / noise_scale
    input_vol = np.maximum(input_vol, 0)  # No negativo
    
    return {
        'synthetic_1M': {
            'input': input_vol,
            'target': target_vol
        }
    }

def simulate_different_models(input_vol, target_vol):
    """Simula diferentes tipos de modelos para comparar"""
    
    models = {}
    
    # 1. Modelo TRIVIAL (solo multiplicador)
    scale_factor = np.mean(target_vol[target_vol > 0.01 * target_vol.max()] / 
                          input_vol[target_vol > 0.01 * target_vol.max()])
    models['trivial'] = input_vol * scale_factor
    
    # 2. Modelo INTELIGENTE (con denoising espacial)
    from scipy import ndimage
    # Suavizado adaptativo + corrección local
    smoothed = ndimage.gaussian_filter(input_vol, sigma=1.0)
    local_correction = 0.8 * input_vol + 0.2 * smoothed
    models['intelligent'] = local_correction * (scale_factor * 0.95)
    
    # 3. Modelo SEMI-INTELIGENTE (multiplicador + poco denoising)
    slightly_smoothed = ndimage.gaussian_filter(input_vol, sigma=0.5)
    models['semi_intelligent'] = (0.9 * input_vol + 0.1 * slightly_smoothed) * scale_factor
    
    return models

def analyze_model_behavior(input_vol, target_vol, pred_vol, model_name="Unknown"):
    """Analiza si un modelo es solo un multiplicador trivial"""
    
    print(f"\n📊 ANALIZANDO: {model_name}")
    print("="*50)
    
    # Máscara para voxels significativos (>1% del máximo)
    mask = target_vol > 0.01 * target_vol.max()
    
    if mask.sum() == 0:
        return None
    
    # Datos enmascarados
    input_masked = input_vol[mask]
    target_masked = target_vol[mask]
    pred_masked = pred_vol[mask]
    
    # Evitar división por cero
    valid_mask = input_masked > 1e-10
    input_valid = input_masked[valid_mask]
    target_valid = target_masked[valid_mask]
    pred_valid = pred_masked[valid_mask]
    
    if len(input_valid) < 100:  # Muy pocos puntos
        print("   ⚠️ Insuficientes datos válidos")
        return None
    
    # 1. ANÁLISIS DE FACTOR DE ESCALA
    scale_factors = target_valid / input_valid
    mean_scale = np.mean(scale_factors)
    std_scale = np.std(scale_factors)
    cv_scale = std_scale / mean_scale  # Coeficiente de variación
    
    print(f"   Factor escala: μ={mean_scale:.2f}, σ={std_scale:.2f}, CV={cv_scale:.3f}")
    
    # 2. PREDICCIÓN INGENUA (multiplicador simple)
    naive_pred = input_vol * mean_scale
    naive_valid = naive_pred[mask][valid_mask]
    
    # 3. MÉTRICAS DE COMPARACIÓN
    mae_pred = np.mean(np.abs(pred_valid - target_valid))
    mae_naive = np.mean(np.abs(naive_valid - target_valid))
    mae_pred_naive = np.mean(np.abs(pred_valid - naive_valid))
    
    # Correlaciones
    corr_pred_target = np.corrcoef(pred_valid, target_valid)[0, 1]
    corr_naive_target = np.corrcoef(naive_valid, target_valid)[0, 1]
    corr_pred_naive = np.corrcoef(pred_valid, naive_valid)[0, 1]
    
    # 4. ANÁLISIS DE TRIVIALIDAD
    improvement_factor = mae_naive / mae_pred if mae_pred > 0 else 1
    
    print(f"   MAE vs target:    Modelo={mae_pred:.4f}, Naïve={mae_naive:.4f}")
    print(f"   Correlaciones:    Modelo-Target={corr_pred_target:.4f}")
    print(f"                     Naïve-Target={corr_naive_target:.4f}")
    print(f"                     Modelo-Naïve={corr_pred_naive:.4f}")
    print(f"   Mejora vs Naïve:  {improvement_factor:.2f}x")
    
    # 5. VEREDICTO
    is_trivial_threshold = 0.98
    significant_improvement = 1.5
    
    if corr_pred_naive > is_trivial_threshold:
        verdict = "⚠️ MULTIPLICADOR TRIVIAL"
        explanation = "Alta correlación con multiplicador simple"
        useful = False
    elif improvement_factor < significant_improvement:
        verdict = "❌ POCO ÚTIL"
        explanation = f"Mejora insuficiente ({improvement_factor:.1f}x)"
        useful = False
    elif improvement_factor > 3.0:
        verdict = "✅ MUY INTELIGENTE"
        explanation = f"Excelente mejora ({improvement_factor:.1f}x)"
        useful = True
    else:
        verdict = "🔶 MODERADAMENTE ÚTIL"
        explanation = f"Mejora moderada ({improvement_factor:.1f}x)"
        useful = True
    
    print(f"   🎯 VEREDICTO: {verdict}")
    print(f"      Razón: {explanation}")
    
    return {
        'model_name': model_name,
        'mean_scale': float(mean_scale),
        'cv_scale': float(cv_scale),
        'mae_pred': float(mae_pred),
        'mae_naive': float(mae_naive),
        'corr_pred_target': float(corr_pred_target),
        'corr_pred_naive': float(corr_pred_naive),
        'improvement_factor': float(improvement_factor),
        'verdict': verdict,
        'is_useful': useful
    }

def create_visualization(input_vol, target_vol, models_dict, output_dir):
    """Crea visualización comparativa"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Seleccionar slice con más información
    slice_doses = [target_vol[z].sum() for z in range(target_vol.shape[0])]
    best_z = np.argmax(slice_doses)
    
    n_models = len(models_dict)
    fig, axes = plt.subplots(2, n_models + 2, figsize=(4*(n_models + 2), 8))
    
    vmax = target_vol.max()
    
    # Fila superior: Volúmenes
    axes[0,0].imshow(input_vol[best_z], cmap='hot', vmin=0, vmax=vmax, aspect='auto')
    axes[0,0].set_title('Input (1M eventos)', fontsize=12, fontweight='bold')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(target_vol[best_z], cmap='hot', vmin=0, vmax=vmax, aspect='auto')
    axes[0,1].set_title('Target (29M eventos)', fontsize=12, fontweight='bold')
    axes[0,1].axis('off')
    
    # Modelos
    for i, (model_name, pred_vol) in enumerate(models_dict.items()):
        col = i + 2
        axes[0,col].imshow(pred_vol[best_z], cmap='hot', vmin=0, vmax=vmax, aspect='auto')
        axes[0,col].set_title(f'Modelo: {model_name}', fontsize=12)
        axes[0,col].axis('off')
    
    # Fila inferior: Errores vs Target
    axes[1,0].text(0.5, 0.5, 'Input vs Target\nError Maps', 
                   ha='center', va='center', fontsize=12, fontweight='bold')
    axes[1,0].axis('off')
    
    error_input = np.abs(input_vol[best_z] - target_vol[best_z])
    axes[1,1].imshow(error_input, cmap='viridis', aspect='auto')
    axes[1,1].set_title('Error: Input vs Target', fontsize=12)
    axes[1,1].axis('off')
    
    for i, (model_name, pred_vol) in enumerate(models_dict.items()):
        col = i + 2
        error_pred = np.abs(pred_vol[best_z] - target_vol[best_z])
        axes[1,col].imshow(error_pred, cmap='viridis', aspect='auto')
        axes[1,col].set_title(f'Error: {model_name}', fontsize=12)
        axes[1,col].axis('off')
    
    fig.suptitle(f'Análisis de Multiplicador Trivial (Slice z={best_z})', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plot_file = output_dir / 'trivial_multiplier_analysis.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_file

def main():
    parser = argparse.ArgumentParser(description='Verificar si modelo es multiplicador trivial')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Ruta al modelo entrenado (.pt)')
    parser.add_argument('--data-dir', type=str, default='exports',
                        help='Directorio con datos exportados')
    parser.add_argument('--output-dir', type=str, default='trivial_analysis',
                        help='Directorio de salida')
    parser.add_argument('--synthetic', action='store_true',
                        help='Usar datos sintéticos para demonstración')
    
    args = parser.parse_args()
    
    print("="*80)
    print("🔍 VERIFICACIÓN: ¿MULTIPLICADOR TRIVIAL?")
    print("="*80)
    print("Hipótesis nula: El modelo solo hace input × factor_constante = predicción")
    print("Si fuera cierto, sería inútil vs simular más eventos directamente\n")
    
    # Crear directorio de salida
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Obtener datos (reales o sintéticos)
    if args.synthetic or not args.model_path:
        print("📦 Usando datos sintéticos para demonstración...")
        data_cases = create_synthetic_data()
    else:
        print("📁 Intentando cargar datos reales...")
        # TODO: Implementar carga de datos reales desde exports/
        print("❌ Carga de datos reales no implementada aún")
        return
    
    all_results = []
    
    # Procesar cada caso
    for case_name, case_data in data_cases.items():
        input_vol = case_data['input']
        target_vol = case_data['target']
        
        print(f"\n🔍 PROCESANDO: {case_name}")
        print(f"   Shape: {input_vol.shape}")
        print(f"   Input range: [{input_vol.min():.3f}, {input_vol.max():.3f}]")
        print(f"   Target range: [{target_vol.min():.3f}, {target_vol.max():.3f}]")
        
        # Generar diferentes tipos de modelos para comparar
        models_dict = simulate_different_models(input_vol, target_vol)
        
        # Analizar cada modelo
        case_results = []
        for model_name, pred_vol in models_dict.items():
            result = analyze_model_behavior(input_vol, target_vol, pred_vol, model_name)
            if result:
                case_results.append(result)
        
        # Visualización
        plot_file = create_visualization(input_vol, target_vol, models_dict, output_dir)
        print(f"\n✓ Visualización guardada: {plot_file}")
        
        all_results.extend(case_results)
    
    # Resumen final
    print("\n" + "="*80)
    print("📊 RESUMEN GLOBAL")
    print("="*80)
    
    if not all_results:
        print("❌ No se pudieron analizar casos")
        return
    
    trivial_models = [r for r in all_results if 'TRIVIAL' in r['verdict']]
    useful_models = [r for r in all_results if r['is_useful']]
    
    print(f"Total modelos analizados: {len(all_results)}")
    print(f"Modelos triviales: {len(trivial_models)}")
    print(f"Modelos útiles: {len(useful_models)}")
    
    print("\n🎯 INTERPRETACIÓN:")
    if len(trivial_models) > len(all_results) // 2:
        print("⚠️  La mayoría de modelos son multiplicadores triviales")
        print("   → No aportan más que simular más eventos")
        print("   → Revisar arquitectura y función de pérdida")
    else:
        print("✅ Los modelos muestran aprendizaje inteligente")
        print("   → Van más allá del simple escalado")
        print("   → Útiles para denoising de dosis")
    
    # Guardar resultados
    results_file = output_dir / 'trivial_analysis_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'summary': {
                'total_models': len(all_results),
                'trivial_models': len(trivial_models),
                'useful_models': len(useful_models)
            },
            'detailed_results': all_results
        }, f, indent=2)
    
    print(f"\n✓ Resultados guardados: {results_file}")

if __name__ == "__main__":
    try:
        import scipy.ndimage
    except ImportError:
        print("⚠️ scipy no disponible, usando denoising básico")
    
    main()