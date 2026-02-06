# ¿Por Qué v3 NO Predice Constante?

## El Problema de v1: Matemática vs Física

### v1: Loss Estándar MSE
```python
loss = MSE(pred, target) = mean((pred - target)²)
```

**¿Qué hace el optimizer?** Minimizar el error cuadrático promedio.

**Escenario de v1**:
- 96% de voxeles: target ≈ 0 (ruido del fondo)
- 4% de voxeles: target ∈ [0, 1000] (señal real)

**Conclusión trivial del modelo**:
```
Si predigo "0" en todas partes:
  MSE ≈ 0²×0.96 + (1000²×0.04) / 100 
     ≈ 4,000
  
Si predigo la estructura real:
  MSE ≈ (error_core)² × 0.04 + (error_low)² × 0.96
  
Para minimizar MSE global, el modelo ELIGE predecir ~0
porque el término 0.96 domina.
```

**Resultado**: Predicción plana (~60 = media global)

---

## v3: Loss Exponencial = Corrección Matemática

### Cambio Fundamental
```python
# v1: Trata todos los errores igual
loss = mean(error²)

# v3: Da más peso a errores donde hay dosis
weights = exp(dose/ref) - 1  # Exponencial
loss = mean(error² × weights) / mean(weights)
```

### Ejemplo Numérico

Imaginemos 100 voxeles:
- Voxel 1: dose=1000 (core)  → weight ≈ 7.4
- Voxeles 2-100: dose=0 (ruido) → weight ≈ 0

**Si el modelo predice constante 60:**
```
error_core = |60 - 1000|² = 940² ≈ 883,600
weighted_error_core = 883,600 × 7.4 ≈ 6,538,640

error_noise = |60 - 0|² = 60² = 3,600  
weighted_error_noise = 3,600 × 0 ≈ 0 (nearly)

total_weighted_loss ≈ 6,538,640  (ALTÍSIMO)
```

**Si el modelo predice la estructura real:**
```
error_core = |1000 - 1000|² = 0
weighted_error_core = 0 × 7.4 = 0

error_noise = |0 - 0|² = 0  
weighted_error_noise = 0 × 0 = 0

total_weighted_loss ≈ 0  (MÍNIMO)
```

**Conclusión**: El optimizer OBLIGA a aprender la estructura porque el costo de predecir mal el core es exponencialmente alto.

---

## Los Otros 3 Pilares Refuerzan Esto

### Pilar 2: Entrada CT
Sin CT: la red intenta reconstruir dosis solo mirando dosis ruidosa
→ Tarea imposible (noise + noise = noise)

Con CT: la red entiende "aquí hay hueso, debe haber atenuación"
→ Tarea posible (noise + geometría = estructura)

### Pilar 3: Arquitectura Avanzada
- **Residual blocks**: Permite que gradientes de la loss exponencial fluyan a capas profundas sin desvanecerse
- **SE blocks**: Red enfatiza canales importantes donde hay transiciones dosis
- **BatchNorm**: Estabiliza gradientes exponenciales (pueden ser grandes)

### Pilar 4: Muchos Datos
Con 56k volúmenes, el modelo ve toda la variabilidad física real.
Con 80 muestras, los patrones se repiten → data augmentation ayuda

---

## Prueba de Concepto: Juguete Matemático

### Scenario Juguete
Predecir 1D: 10 puntos con dosis [0,0,0,0,0,0,0,1000,1000,1000]

**v1 (MSE estándar)**:
```
Si pred = [70, 70, 70, 70, 70, 70, 70, 70, 70, 70]:
  MSE = (70²×7 + (70-1000)²×3) / 10
      = (34,300 + 2,586,300) / 10
      = 262,060  ← Pero para el optimizer es "OK"
  
  Porque 70% del error es en voxeles con peso bajo (ruido)
```

**v3 (Loss exponencial)**:
```
ref_dose = 1000 × 0.5 = 500
weights = [0.1, ..., 0.1, 1.0, 1.0, 1.0]

Si pred = [70, ..., 70]:
  weighted_loss = (70² × 0.1×7) + (930²×1.0×3)
                ≈ 3,500 + 2,593,500
                = 2,597,000  ← ENORME
  
Si pred = [0, ..., 0, 1000, 1000, 1000]:
  weighted_loss ≈ 0  ← MÍNIMO
```

**Conclusión**: v3 matemáticamente OBLIGA al modelo a aprender.

---

## Garantías de v3

1. ✅ **No predice constante**: El cost de mal-predecir el core crece exponencialmente
2. ✅ **Prioriza precisión en core**: Donde importa clínicamente (>20% de dosis máx)
3. ✅ **Tolera error en ruido**: Bajo weight en voxeles con dosis baja
4. ✅ **Escalable**: Los hiperparámetros (ref_dose_percentile) se pueden tunar

---

## Comparativa Visual (Esperada)

```
v1 Output:        v3 Output:
    |                 |
 60 |------|       100 |    ╱╲
    |      |        80 |   ╱  ╲
 40 |      |        60 |  ╱    ╲___
    |      |        40 |_╱
 20 |      |        20 |
    |      |         0 |________________
 0  |______|
    (PLANA)        (FORMA REALISTA)
```

---

## Resumen: La Clave

**v1**: MSE trata el error global → 96% son ceros → modelo aprende "predecir cero"

**v3**: Loss exponencial enfatiza el 4% con estructura → modelo aprende "predecir estructura"

Es puramente un **cambio en la función objetivo del optimizer**. El modelo es capaz de aprender, solo necesitaba incentivos correctos.

