# 📊 VALIDACIÓN DE GROUND TRUTH (29.4M PARTICLES)

## Resultado del Análisis Local

```
Shape: (300, 150, 150)
Min: 0.0, Max: 979.2
```

### A. ✅ PDD (SUAVIDAD): EXCELENTE
- **PDD Smoothness: 1.6354**
- Interpretación: Entre 0.5-2 = **OK, aceptable**
- El área está apenas fuera de "muy suave" (<0.5) pero **NO es ruidosa**
- La curva tiene estructura clara: crece hasta d_max (índice 13), luego decae exponencial
- **Veredicto: ACEPTABLE para training** ✅

### B. ✅ PERFILES TRANSVERSALES (SIMETRÍA): EXCELENTE
- **Asimetría X: 1.1584%** (< 5% = simétrico) ✅
- **Asimetría Y: 1.2645%** (< 5% = simétrico) ✅
- Esto significa: Ambos lados del haz son prácticamente idénticos
- NO hay sesgos estadísticos que confundan al modelo
- **Veredicto: EXCELENTE** ✅

### C. ⚠️ SNR EN PERIFERIA: DÉBIL
- **SNR: 0.2516** (bajo, < 1)
- **Media periferia: 1.18**, **Std: 4.70**
- Significa: El ruido en la periferia es **3.97x más grande que la señal**
- Hay "hot pixels" aislados fuera del haz principal
- **Señal de alarma**: La red podría aprender ruido como patrón

## 🎯 CONCLUSIÓN: 29.4M ES SUFICIENTE PERO CON LIMITACIONES

### ✅ Fortalezas:
1. **PDD muy suave** → Model puede entender ley de atenuación
2. **Perfiles totalmente simétricos** → Sin sesgos direccionales
3. **Estadística general buena** → Las 29.4M partículas producen distribución coherente

### ⚠️ Limitaciones:
1. **Periferia ruidosa** → Puntos calientes aislados fuera del haz
   - Esto NO afecta el aprendizaje del núcleo (beam core)
   - PODRÍA confundir a la red si la red es muy sensible a outliers
   
## 💡 RECOMENDACIONES

### Opción A: Usar 29.4M COMO IS
- **PRO:** Más rápido, ya tienes los datos
- **CON:** Red podría aprender a predecir ruido periférico

### Opción B: Aumentar a ~40-50M partículas
- **PRO:** Reduce ruido periférico (statistical smoothing)
- **CON:** Simulaciones más largas
- **Estimado:** +50% tiempo de simulación

### Opción C: Post-procesar 29.4M (RECOMENDADO)
```python
# Aplicar filtro suave en periferia para eliminar hot pixels
dose_smooth = gaussian_filter(dose, sigma=0.5)
# O: threshold suave en regiones donde dose < 10% max
```

## 🔍 ANÁLISIS ESPECÍFICO DEL PROBLEMA ANTERIOR

Tu modelo anterior predecía pred_max ≈ 0.54 constantemente.

Con estos datos (29.4M):
- **Target Max varía: 2.7x - 21.7x** (según input level)
- **PDD es muy suave** → Network PUEDE aprender esta variación
- **Perfiles son simétricos** → NO hay sesgos que confundan
- **Ruido periférico bajo (SNR=0.25)** → ⚠️ Posible problema

### El ruido periférico podría ser la razón de que la red "colapsa":
1. Durante training, la red ve ruido aleatorio en periferia
2. La red aprende que la periferia es impredecible
3. Como entrada a downsampling, esto podría contaminar features globales
4. Red "se rinde" y predice constante para evitar errores periféricos

## 🚀 PRÓXIMOS PASOS

1. **Opción 1 (Rápido):** Entrena COMO IS con 29.4M, pero:
   - Usa MAE loss en lugar de MSE (menos penaliza outliers periféricos)
   - O: Usa weighted loss (penaliza menos la periferia)

2. **Opción 2 (Más Robusto):** Post-procesa targets:
   ```python
   from scipy.ndimage import gaussian_filter
   dose_smooth = gaussian_filter(dose, sigma=0.5)
   # Blend: donde dose es baja, interpola smoothly
   ```

3. **Opción 3 (Experimental):** Las 29.4M son BUENAS para el núcleo.
   Considerá **aumentar lentamente a 40M** si las sims son rápidas en cluster.

---

**RECOMENDACIÓN FINAL:** Usa 29.4M PERO aplica post-procesamiento suave en periferia
para eliminar hot pixels aislados. Esto mantiene la buena estadística del núcleo
mientras reduce el ruido que podría confundir a la red.
