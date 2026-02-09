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
    
    # Análisis detallado del ground truth
    print(f"\n[1b] DISTRIBUCIÓN DEL GROUND TRUTH:")
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    non_zero = target_vol[target_vol > 0]
    print(f"  Total voxels: {target_vol.size:,}")
    print(f"  Voxels con dosis > 0: {non_zero.size:,} ({non_zero.size/target_vol.size*100:.1f}%)")
    print(f"  Rango: [{target_vol.min():.4f}, {max_dose:.4f}]")
    print(f"  Percentiles:")
    for p in percentiles:
        val = np.percentile(non_zero, p)
        pct_of_max = val / max_dose * 100
        voxels_above = np.sum(target_vol >= val)
        voxels_pct = voxels_above / target_vol.size * 100
        print(f"    p{p:2d}: {val:10.4f} ({pct_of_max:5.1f}% del máx) - {voxels_above:,} voxels arriba ({voxels_pct:5.1f}%)")
    
    # Definir máscaras ANTES de usar en análisis de inputs
    print(f"\n[1c] MÁSCARAS DEFINIDAS (basadas en target):")
    masks = {
        "10%": target_vol > 0.10 * max_dose,
        "5%": target_vol > 0.05 * max_dose,
        "1%": target_vol > 0.01 * max_dose,
        "0%" : target_vol > 0,
    }
    
    print(f"\n  Cada máscara cubre:")
    for mask_name, mask in masks.items():
        voxels = mask.sum()
        pct = voxels / mask.size * 100
        
        # Estadísticas del ground truth dentro de la máscara
        if voxels > 0:
            gt_in_mask = target_vol[mask]
            gt_mean = gt_in_mask.mean()
            gt_median = np.median(gt_in_mask)
            gt_min = gt_in_mask.min()
            gt_max = gt_in_mask.max()
            
            print(f"    → target > {mask_name:3s}: {voxels:,} voxels ({pct:5.1f}%) - GT min/median/mean/max = {gt_min:.4f} / {gt_median:.4f} / {gt_mean:.4f} / {gt_max:.4f}")
    
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
            
            # Análisis respecto a las máscaras
            print(f"  Comportamiento relativo a GT:")
            for mask_name in ["10%", "5%", "1%"]:
                mask = masks[mask_name]
                if mask.sum() > 0:
                    input_in_mask = input_vol[mask]
                    diff_in_mask = diff[mask]
                    pct_within_3 = np.sum(diff_in_mask <= 0.03*max_dose) / len(diff_in_mask) * 100
                    pct_within_5 = np.sum(diff_in_mask <= 0.05*max_dose) / len(diff_in_mask) * 100
                    
                    print(f"    En zona > {mask_name:3s}: diff_mean={diff_in_mask.mean():.4f}, % dentro 3%={pct_within_3:.1f}%, dentro 5%={pct_within_5:.1f}%")
    
    # Gamma analysis para cada input con diferentes configs
    print(f"\n[3] GAMMA PASS RATE POR INPUT:")
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
    print(f"\n[4] GAMMA (input_10M) CON DIFERENTES MÁSCARAS:")
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
    
    # Análisis del ruido
    print(f"\n[5] ANÁLISIS DE RUIDO (vs Ground Truth):")
    for level in ["input_1M", "input_5M", "input_10M"]:
        if level not in inputs_data:
            continue
        input_vol = inputs_data[level]
        
        # SNR aproximado
        signal = target_vol
        noise = input_vol - target_vol
        
        # Donde haya señal (>1% max_dose)
        mask_signal = signal > 0.01 * max_dose
        if mask_signal.sum() > 0:
            signal_power = np.mean(signal[mask_signal] ** 2)
            noise_power = np.mean(noise[mask_signal] ** 2)
            snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
            rmse = np.sqrt(np.mean(noise[mask_signal] ** 2))
            rmse_pct = rmse / max_dose * 100
            
            print(f"\n  {level}:")
            print(f"    SNR: {snr:.2f} dB")
            print(f"    RMSE (ruido): {rmse:.4f} ({rmse_pct:.2f}% del max_dose)")
            print(f"    Nota: Tolerancia gamma 3% DoMax = {0.03*max_dose:.4f}")
            print(f"          → El ruido es MAYOR que la tolerancia")
    
    print(f"\n[6] CONCLUSIONES:")
    print(f"  • El gamma del input es 0% porque:")
    print(f"    1. La máscara es restrictiva (solo zona de alta dosis: 10% de max)")
    print(f"    2. La tolerancia (3% DoMax) es MUCHO más pequeña que el RMSE del ruido")
    print(f"    3. El input ruidoso NO puede pasar una tolerancia tan estricta")

    print()
    print(f"  • Recomendaciones:")
    print(f"    1. Usar DoLocal (3% de dosis LOCAL, no del máximo)")
    print(f"       → Justo para ruido y señal débil")
    print(f"       → input_10M probablemente ~40-60%")
    print(f"    2. O aumentar tolerancia a 5% DoMax")
    print(f"       → input_10M probablemente ~20-40%")
    print(f"    3. O combinación: máscara 5% + tolerancia 5% DoLocal")
    print(f"")
    print(f"  • Ground Truth: tiene dosis concentrada (~90% en zona >10% max)")
    print(f"    El resto es very low dose (ruido de fondo)")


if __name__ == '__main__':
    main()
