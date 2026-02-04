# Job Scripts para Cluster

Esta carpeta contiene templates de jobs SLURM listos para usar.

## 📁 Archivos

- **`slurm_single_job.sh`**: Job individual con análisis automático
- **`slurm_array_job.sh`**: Array job para múltiples simulaciones en paralelo

## 🚀 Uso

### 1. Job Individual

Para una simulación única con muchas partículas:

```bash
# Crear directorio de logs
mkdir -p logs

# Editar parámetros en slurm_single_job.sh si es necesario:
# - N_PARTICLES (default: 1M)
# - INPUT_PHSP (default: Varian_Clinac_2100CD_6MeV_15x15.root)
# - CPUS (default: 16)

# Enviar job
sbatch jobs/slurm_single_job.sh

# Monitorear
squeue -u $USER
tail -f logs/dose_*.out
```

### 2. Array Job (Múltiples Simulaciones)

Para ejecutar 10 simulaciones en paralelo (útil para estadísticas):

```bash
# Crear directorio de logs
mkdir -p logs

# Enviar array job (10 tasks)
sbatch jobs/slurm_array_job.sh

# Monitorear todos los jobs
squeue -u $USER

# Ver progreso de un task específico
tail -f logs/dose_*_3.out  # Task 3
```

Esto ejecutará:
- **10 jobs simultáneos** (si hay recursos)
- **500k partículas por job** = 5M total
- **Seeds diferentes** por job (estadísticas independientes)
- **Análisis automático** al finalizar cada uno

### 3. Personalización

**Cambiar número de tasks en array job:**
```bash
# Editar línea en slurm_array_job.sh:
#SBATCH --array=0-19  # Para 20 tasks
```

**Cambiar partículas por job:**
```bash
# Editar en el script:
N_PARTICLES=1000000  # 1M por job
```

**Usar diferentes archivos de input:**
```bash
# Editar:
INPUT_PHSP="data/IAEA/mi_archivo_custom.root"
```

## 📊 Resultados

Después de ejecutar:

**Single Job:**
```
output_job_123456/          # Datos de simulación
results/analysis_123456/    # Métricas y gráficas
logs/dose_123456.out        # Log stdout
logs/dose_123456.err        # Log stderr
```

**Array Job:**
```
output_array_789_0/         # Task 0
output_array_789_1/         # Task 1
...
output_array_789_9/         # Task 9

results/analysis_789_0/     # Análisis task 0
results/analysis_789_1/     # Análisis task 1
...
```

## 🔧 Troubleshooting

**Job no inicia:**
```bash
# Ver estado detallado
scontrol show job JOBID

# Ver prioridad en cola
sprio -j JOBID
```

**Job falla inmediatamente:**
```bash
# Revisar error log
cat logs/dose_JOBID.err

# Errores comunes:
# - "No such file": Archivo de input no transferido
# - "ModuleNotFoundError": Entorno no activado correctamente
# - "IndexError": Patch de OpenGate no aplicado
```

**Cancelar jobs:**
```bash
# Cancelar un job
scancel JOBID

# Cancelar todos mis jobs
scancel -u $USER

# Cancelar array completo
scancel ARRAY_JOB_ID
```

## ⚡ Tips de Optimización

1. **CPUs**: Usar múltiplos de cores por nodo (8, 16, 32)
2. **Memoria**: 4GB suficiente para 500k-1M partículas, 8GB para 5M+
3. **Tiempo**: Estimar ~1-2 horas por millón de partículas en single-thread
4. **Array vs Single**: 
   - Array: Mejor para explorar parámetros o estadísticas
   - Single: Mejor para simulación única grande

## 📚 Referencias

- [SLURM Array Jobs](https://slurm.schedmd.com/job_array.html)
- [OpenGate Documentation](https://opengate-python.readthedocs.io/)
- Ver `CLUSTER_SETUP.md` en raíz del proyecto para setup inicial
