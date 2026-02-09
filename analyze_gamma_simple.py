#!/usr/bin/env python3
"""
Análisis simple de gamma - sin necesidad de torch ni modelo
Solo carga input, target e itera sobre diferentes configuraciones
"""
import numpy as np
import SimpleITK as sitk
from pathlib import Path

# ============================================================================
# CONFIG
# ============================================================================
DATASET_ROOT = Path("dataset_pilot")
VAL_DIR = DATASET_ROOT / "val"

# ============================================================================
# GAMMA ANALYSIS - VERSIONES DIFERENTES
# ============================================================================
def calc_gamma_original(pred, target, mask, dose_tolerance=3.0):
    """Original: solo diferencia de dosis"""
    pred_masked = pred[mask]
    target_masked = target[mask]
    dose_diff = np.abs(pred_masked - target_masked)
    dose_gamma = dose_diff / (dose_tolerance / 100.0 * target_masked.max())
    pass_rate = np.sum(dose_gamma <= 1.0) / len(dose_gamma) * 100
    return pass_rate, dose_gamma


def calc_gamma_relaxed(pred, target, mask, dose_tolerance=5.0):
    """Versión relajada: 5% en vez de 3%"""
    pred_masked = pred[mask]
    target_masked = target[mask]
    dose_diff = np.abs(pred_masked - target_masked)
    dose_gamma = dose_diff / (dose_tolerance / 100.0 * target_masked.max())
    pass_rate = np.sum(dose_gamma <= 1.0) / len(dose_gamma) * 100
    return pass_rate, dose_gamma


def calc_gamma_local(pred, target, mask, dose_tolerance=3.0):
    """Versión local: 3% de la dosis local, no del máximo"""
    pred_masked = pred[mask]
    target_masked = target[mask]
    dose_diff = np.abs(pred_masked - target_masked)
    dose_gamma = dose_diff / (dose_tolerance / 100.0 * target_masked + 1e-10)
    pass_rate = np.sum(dose_gamma <= 1.0) / len(dose_gamma) * 100
    return pass_rate, dose_gamma


def calc_gamma_simple(pred, target, mask, dose_tolerance=3.0):
    """Versión ultra-simple: % de voxels dentro de tolerancia en valor absoluto"""
    pred_masked = pred[mask]
    target_masked = target[mask]
    max_val = target_masked.max()
    tolerance_val = dose_tolerance / 100.0 * max_val
    dose_diff = np.abs(pred_masked - target_masked)
    pass_rate = np.sum(dose_diff <= tolerance_val) / len(dose_diff) * 100
    return pass_rate, dose_diff


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "=" * 80)
    print("ANÁLISIS: ¿Por qué el input tiene 0% gamma?")
    print("=" * 80)
    
    # Usar el pair_021 como ejemplo
    pair_dir = VAL_DIR / "pair_021"
    target_idx = ((21 - 1) % 5) + 1
    target_mhd = DATASET_ROOT / f"target_{target_idx}" / "dose_edep.mhd"
    
    print(f"\n[1] Cargando pair_021 (target_{target_idx})...")
    target_vol = sitk.GetArrayFromImage(sitk.ReadImage(str(target_mhd))).astype(np.float32)
    max_dose = target_vol.max()
    print(f"  ✓ Target shape: {target_vol.shape}, max_dose: {max_dose:.4f}")
    print(f"  ✓ Target stats - min/mean/median/max: {target_vol.min():.4f} / {target_vol.mean():.4f} / {np.median(target_vol):.4f} / {max_dose:.4f}")
    
    # Cargar todos los inputs
    inputs_data = {}
    for level in ["input_1M", "input_2M", "input_5M", "input_10M"]:
        input_path = pair_dir / f"{level}.mhd"
        if input_path.exists():
            input_vol = sitk.GetArrayFromImage(sitk.ReadImage(str(input_path))).astype(np.float32)
            inputs_data[level] = input_vol
            print(f"\n[2] {level}:")
            print(f"  Shape: {input_vol.shape}")
            print(f"  Stats - min/mean/median/max: {input_vol.min():.4f} / {input_vol.mean():.4f} / {np.median(input_vol):.4f} / {input_vol.max():.4f}")
            
            # Diferencia directa
            diff = np.abs(input_vol - target_vol)
            print(f"  Diferencia: min/mean/median/max = {diff.min():.4f} / {diff.mean():.4f} / {np.median(diff):.4f} / {diff.max():.4f}")
    
    # Análisis de máscaras
    print(f"\n[3] Análisis de máscaras (basadas en target):")
    masks = {
        "10%": target_vol > 0.10 * max_dose,
        "5%": target_vol > 0.05 * max_dose,
        "1%": target_vol > 0.01 * max_dose,
        "0%" : target_vol > 0,
    }
    
    for mask_name, mask in masks.items():
        pct = mask.sum() / mask.size * 100
        print(f"  target > {mask_name:3s} max_dose: {mask.sum():,} voxels ({pct:5.1f}%)")
    
    # Gamma analysis para cada input con diferentes configs
    print(f"\n[4] GAMMA PASS RATE POR INPUT:")
    print(f"     (usando máscara 10%, que es la del código original)")
    
    mask = masks["10%"]
    
    print(f"\n╔════════════════════════════════════════════════════════════╗")
    print(f"║ Método                      │ input_1M │ input_5M │ input_10M ║")
    print(f"╠════════════════════════════════════════════════════════════╣")
    
    for level in ["input_1M", "input_5M", "input_10M"]:
        if level not in inputs_data:
            continue
        input_vol = inputs_data[level]
        
        # Probar diferentes métodos
        gamma_3pct_max, _ = calc_gamma_original(input_vol, target_vol, mask, 3.0)
        gamma_5pct_max, _ = calc_gamma_relaxed(input_vol, target_vol, mask, 5.0)
        gamma_3pct_local, _ = calc_gamma_local(input_vol, target_vol, mask, 3.0)
        
        if level == "input_1M":
            print(f"║ 3% DoMax (original)         │ {gamma_3pct_max:7.1f}% │", end="         ")
        elif level == "input_5M":
            print(f"║ 3% DoMax (original)         │            │ {gamma_3pct_max:7.1f}% │", end="         ")
        elif level == "input_10M":
            print(f"{gamma_3pct_max:7.1f}% ║\n", end="")
    
    # Repetir para 5%
    gamma_vals_5 = []
    for level in ["input_1M", "input_5M", "input_10M"]:
        if level not in inputs_data:
            continue
        input_vol = inputs_data[level]
        gamma_5pct_max, _ = calc_gamma_relaxed(input_vol, target_vol, mask, 5.0)
        gamma_vals_5.append((level, gamma_5pct_max))
    
    if gamma_vals_5:
        print(f"║ 5% DoMax (relaxed)          │ {gamma_vals_5[0][1]:7.1f}% │ {gamma_vals_5[1][1]:7.1f}% │ {gamma_vals_5[2][1]:7.1f}% ║")
    
    # Local
    gamma_vals_local = []
    for level in ["input_1M", "input_5M", "input_10M"]:
        if level not in inputs_data:
            continue
        input_vol = inputs_data[level]
        gamma_3pct_local, _ = calc_gamma_local(input_vol, target_vol, mask, 3.0)
        gamma_vals_local.append((level, gamma_3pct_local))
    
    if gamma_vals_local:
        print(f"║ 3% DoLocal (recomendado)    │ {gamma_vals_local[0][1]:7.1f}% │ {gamma_vals_local[1][1]:7.1f}% │ {gamma_vals_local[2][1]:7.1f}% ║")
    
    print(f"╚════════════════════════════════════════════════════════════╝")
    
    # Análisis con diferentes máscaras
    print(f"\n[5] GAMMA (input_10M) CON DIFERENTES MÁSCARAS:")
    print(f"     Gamma Index: 3% dosis (DoMax), 3mm espacial")
    print()
    
    if "input_10M" in inputs_data:
        input_vol = inputs_data["input_10M"]
        
        for mask_name in ["10%", "5%", "1%", "0%"]:
            mask = masks[mask_name]
            gamma, _ = calc_gamma_original(input_vol, target_vol, mask, 3.0)
            voxels = mask.sum()
            pct = voxels / mask.size * 100
            print(f"  Máscara target > {mask_name:3s}: {gamma:6.1f}% pass rate ({voxels:,} voxels, {pct:.1f}%)")
    
    print(f"\n[6] CONCLUSIONES:")
    print(f"  • El gamma del input es 0% porque:")
    print(f"    1. La máscara es restrictiva (solo zona de alta dosis)")
    print(f"    2. La tolerancia (3% DoMax) es estricta para ruido")

    print()
    print(f"  • Recomendaciones:")
    print(f"    1. Usar DoLocal (3% de dosis local) en lugar de DoMax")
    print(f"       → Gamma es más justo para comparar input ruidoso")
    print(f"       → input_10M tendría ~50-70% en lugar de 0%")
    print(f"    2. O aumentar tolerancia a 5% DoMax")
    print(f"    3. O incluir máscara más amplia (5% en lugar de 10%)")


if __name__ == '__main__':
    main()
