#!/usr/bin/env python3
"""
Validación de física - Comparar resultados de simulación con teoría
para haces de electrones de 6 MeV.
"""

import json
import sys
from pathlib import Path

print("="*70)
print("VALIDACIÓN DE FÍSICA - HAZ DE ELECTRONES 6 MeV")
print("="*70)

# Valores teóricos esperados para 6 MeV en agua (TG-51, Khan's Physics)
THEORETICAL = {
    "Zmax_mm": 13.0,  # Profundidad de dosis máxima
    "R50_mm": 26.0,   # Rango práctico (E0/2 en MeV)
    "Rp_mm": 30.0,    # Rango proyectado
    "field_size_mm": 150.0  # Campo 15x15 cm
}

# Tolerancias (%)
TOLERANCES = {
    "Zmax_mm": 15,  # ±15% es aceptable
    "R50_mm": 15,   # ±15%
    "FWHM_lateral": 10  # ±10% para tamaño de campo
}

def validate_metric(name, measured, expected, tolerance_pct):
    """Valida una métrica contra valor esperado."""
    if expected == 0:
        return True, 0.0
    
    error_pct = abs(measured - expected) / expected * 100
    passed = error_pct <= tolerance_pct
    
    status = "✅" if passed else "❌"
    print(f"\n{status} {name}:")
    print(f"   Medido:   {measured:.1f} mm")
    print(f"   Esperado: {expected:.1f} mm")
    print(f"   Error:    {error_pct:.1f}% (tolerancia: ±{tolerance_pct}%)")
    
    return passed, error_pct

def main():
    # Cargar resultados de simulación
    results_file = Path("output_phsp_500k/analysis_results.json")
    
    if not results_file.exists():
        print(f"\n❌ ERROR: No se encuentra {results_file}")
        print("Ejecutar primero:")
        print("  python simulations/dose_phsp_500k.py")
        print("  python simulations/analyze_dose_profile.py")
        return 1
    
    print(f"\n📂 Cargando resultados: {results_file}")
    with open(results_file) as f:
        results = json.load(f)
    
    print(f"\n📊 Resultados de simulación (500k partículas):")
    for key, val in results.items():
        if isinstance(val, float):
            print(f"   {key}: {val:.2f}")
        else:
            print(f"   {key}: {val}")
    
    # Validaciones
    print(f"\n{'='*70}")
    print("COMPARACIÓN CON TEORÍA")
    print(f"{'='*70}")
    
    all_passed = True
    errors = {}
    
    # 1. Zmax (profundidad de dosis máxima)
    zmax = results.get('zmax_rel_mm', 0)
    passed, error = validate_metric(
        "Zmax (Profundidad de dosis máxima)",
        zmax,
        THEORETICAL['Zmax_mm'],
        TOLERANCES['Zmax_mm']
    )
    all_passed = all_passed and passed
    errors['Zmax'] = error
    
    # 2. R50 (rango práctico)
    r50 = results.get('r50_rel_mm', 0)
    passed, error = validate_metric(
        "R50 (Rango práctico)",
        r50,
        THEORETICAL['R50_mm'],
        TOLERANCES['R50_mm']
    )
    all_passed = all_passed and passed
    errors['R50'] = error
    
    # Nota sobre R50
    if r50 > THEORETICAL['R50_mm'] * 1.1:
        print(f"   ⚠️  R50 alto puede deberse al filtro E>5.5 MeV")
        print(f"       (excluye electrones de baja energía)")
    
    # 3. FWHM lateral (opcional, si está disponible)
    if 'fwhm_mm' in results:
        fwhm = results['fwhm_mm']
        if fwhm > 0:  # Si se calculó
            passed, error = validate_metric(
                "FWHM lateral (Tamaño de campo)",
                fwhm,
                THEORETICAL['field_size_mm'],
                TOLERANCES['FWHM_lateral']
            )
            # No afecta el resultado global (puede ser del eje Z)
            errors['FWHM'] = error
    
    # Resumen
    print(f"\n{'='*70}")
    print("RESUMEN DE VALIDACIÓN")
    print(f"{'='*70}")
    
    if all_passed:
        print("\n✅ FÍSICA VALIDADA CORRECTAMENTE")
        print("\nLa simulación reproduce correctamente:")
        print("  • Profundidad de dosis máxima (Zmax)")
        print("  • Rango práctico de electrones (R50)")
        print("  • Comportamiento físico esperado para 6 MeV")
        print("\n👍 Listo para ejecutar en cluster")
        return 0
    else:
        print("\n❌ VALIDACIÓN FALLIDA")
        print("\nRevisar:")
        print("  • Geometría del fantoma (agua, densidad correcta)")
        print("  • Lista de física (QGSP_BIC_EMZ)")
        print("  • Phase space source (posición, dirección)")
        print("  • Filtros de energía aplicados")
        print("\n⚠️  Corregir antes de cluster")
        return 1

if __name__ == '__main__':
    try:
        exit(main())
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
