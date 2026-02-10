#!/usr/bin/env python3
"""
Script de diagnóstico para verificar que el código del gamma pass rate funciona
"""
import numpy as np

def calc_gamma_pass_rate(pred, target, mask, dose_tolerance=3.0, distance_tolerance=3.0):
    pred_masked = pred[mask]
    target_masked = target[mask]
    dose_diff = np.abs(pred_masked - target_masked)
    dose_gamma = dose_diff / (dose_tolerance / 100.0 * target_masked.max())
    gamma_values = dose_gamma
    pass_rate = np.sum(gamma_values <= 1.0) / len(gamma_values) * 100
    return pass_rate

def calc_advanced_metrics(pred, target, max_dose, input_vol=None):
    """Test de la función con gamma para input y pred"""
    results = {}
    
    # Gamma Analysis
    mask_significant = target > 0.10 * max_dose
    if mask_significant.sum() > 0:
        gamma_pass_rate = calc_gamma_pass_rate(pred, target, mask_significant)
        results['gamma_pass_rate_%'] = float(gamma_pass_rate)
        
        print(f"DEBUG: input_vol is None? {input_vol is None}")
        if input_vol is not None:
            print(f"DEBUG: Calculando gamma para input...")
            gamma_pass_rate_input = calc_gamma_pass_rate(input_vol, target, mask_significant)
            results['gamma_pass_rate_input_%'] = float(gamma_pass_rate_input)
            print(f"DEBUG: gamma_input = {gamma_pass_rate_input:.1f}%")
        else:
            print(f"DEBUG: input_vol es None, no se calcula gamma_input")
    else:
        results['gamma_pass_rate_%'] = 0.0
        if input_vol is not None:
            results['gamma_pass_rate_input_%'] = 0.0
    
    return results

# Test con datos simulados
print("="*70)
print("TEST DE GAMMA PASS RATE")
print("="*70)

# Crear volúmenes de prueba
target = np.random.rand(100, 100, 100) * 1000
input_vol = target + np.random.randn(100, 100, 100) * 50  # Input ruidoso
pred = target + np.random.randn(100, 100, 100) * 20  # Predicción más limpia

max_dose = target.max()

print(f"\nVolúmenes creados:")
print(f"  target.shape = {target.shape}, max = {target.max():.2f}")
print(f"  input.shape = {input_vol.shape}, max = {input_vol.max():.2f}")
print(f"  pred.shape = {pred.shape}, max = {pred.max():.2f}")

# Llamar a la función con input_vol
print(f"\nLlamando calc_advanced_metrics con input_vol...")
metrics = calc_advanced_metrics(pred, target, max_dose, input_vol=input_vol)

print(f"\nResultados:")
print(f"  gamma_pass_rate_% = {metrics.get('gamma_pass_rate_%', 'NOT FOUND')}")
print(f"  gamma_pass_rate_input_% = {metrics.get('gamma_pass_rate_input_%', 'NOT FOUND')}")

# Simular el print del resumen
adv = metrics
if adv.get('gamma_pass_rate_%', 0) > 0:
    gamma_input = adv.get('gamma_pass_rate_input_%', 0)
    gamma_pred = adv['gamma_pass_rate_%']
    print(f"\nOutput esperado en resumen:")
    if gamma_input > 0:
        print(f"  Gamma pass rate: input={gamma_input:.1f}% → pred={gamma_pred:.1f}% (mejora: +{gamma_pred-gamma_input:.1f}%)")
    else:
        print(f"  Gamma pass rate: {gamma_pred:.1f}%")
        print(f"  ⚠ gamma_input es 0, algo falló")

print("\n" + "="*70)
print("Si ves 'gamma_pass_rate_input_%' y una mejora, el código funciona.")
print("Si no, hay un problema con el archivo evaluate_model.py en el cluster.")
print("="*70)
