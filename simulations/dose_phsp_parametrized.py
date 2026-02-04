#!/usr/bin/env python3
"""
Simulación parametrizada con PhaseSpaceSource para ejecución en cluster.

Uso:
    python dose_phsp_parametrized.py \
        --input data/IAEA/phsp_500k.root \
        --output output_job_001 \
        --n-particles 500000 \
        --threads 8 \
        --seed 123
"""

import argparse
import opengate as gate
import numpy as np
from pathlib import Path
import os
import json
import sys

# Backend sin display
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import matplotlib
matplotlib.use('Agg')


def run_simulation(args):
    """Ejecuta la simulación con los parámetros dados."""
    
    output_folder = Path(args.output)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print(f"SIMULACIÓN PHSP - JOB {args.job_id if args.job_id else 'LOCAL'}")
    print("="*70)
    print(f"  Input:     {args.input}")
    print(f"  Output:    {output_folder}")
    print(f"  Particles: {args.n_particles:,}")
    print(f"  Threads:   {args.threads}")
    print(f"  Seed:      {args.seed}")
    print("="*70)
    
    # Verificar que existe el archivo de entrada
    if not Path(args.input).exists():
        print(f"❌ ERROR: No se encuentra {args.input}")
        sys.exit(1)
    
    # Crear simulación
    sim = gate.Simulation()
    sim.random_seed = args.seed
    sim.number_of_threads = args.threads
    sim.output_dir = str(output_folder)
    
    # Física
    sim.physics_manager.physics_list_name = 'QGSP_BIC_EMZ'
    
    # Geometría
    world = sim.world
    world.size = [100, 100, 150]
    world.material = 'G4_AIR'
    
    water_log = sim.add_volume('Box', 'water')
    water_log.size = [100, 100, 30]
    water_log.translation = [0, 0, 15]
    water_log.material = 'G4_WATER'
    water_log.color = [0, 0.6, 0.8, 0.5]
    
    print(f"\n🔧 Geometría:")
    print(f"   Mundo: 100x100x150 cm (aire)")
    print(f"   Agua:  100x100x30 cm @ Z=15cm")
    
    # PhaseSpaceSource
    source = sim.add_source('PhaseSpaceSource', 'root_phsp')
    source.phsp_file = args.input
    source.particle = 'e-'
    source.n = args.n_particles
    
    # Configuración explícita de keys para ROOT
    source.position_key_x = 'x'
    source.position_key_y = 'y'
    source.position_key_z = 'z'
    source.direction_key_x = 'dx'
    source.direction_key_y = 'dy'
    source.direction_key_z = 'dz'
    source.energy_key = 'E'
    source.weight_key = 'w'
    source.PDGCode_key = 'pid'
    
    # entry_start para multi-thread
    if args.threads > 1:
        # Distribuir el phase space entre threads
        entries_per_thread = args.n_particles // args.threads
        source.entry_start = [i * entries_per_thread for i in range(args.threads)]
        print(f"   entry_start: {source.entry_start[:4]}... ({args.threads} threads)")
    else:
        source.entry_start = 0
    
    print(f"\n📊 PhaseSpaceSource:")
    print(f"   File: {source.phsp_file}")
    print(f"   N:    {source.n:,}")
    
    # Actor de dosis en profundidad (Z)
    dose_z = sim.add_actor('DoseActor', 'dose_z')
    dose_z.attached_to = 'water'
    dose_z.size = [1, 1, 300]  # 1x1x300 voxels
    dose_z.spacing = [100, 100, 1]  # 100x100x1 mm
    dose_z.output_filename = 'dose_z.mhd'
    dose_z.write_to_disk = True
    
    # Actor de dosis transversal (XY) en Zmax
    if args.compute_xy:
        dose_xy = sim.add_actor('DoseActor', 'dose_xy')
        dose_xy.attached_to = 'water'
        dose_xy.size = [200, 200, 1]  # 200x200x1 voxels
        dose_xy.spacing = [1, 1, 300]  # 1x1x300 mm (todo el espesor)
        dose_xy.output_filename = 'dose_xy.mhd'
        dose_xy.write_to_disk = True
        print(f"   + DoseActor XY: 200x200x1 @ 1mm")
    
    # Estadísticas
    stats_actor = sim.add_actor('SimulationStatisticsActor', 'stats')
    
    # Ejecutar
    print(f"\n🚀 Ejecutando simulación...")
    try:
        sim.run()
        print(f"✅ Simulación completada")
    except Exception as e:
        print(f"❌ ERROR en simulación: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Post-procesamiento básico
    try:
        stats = sim.get_actor('stats')
        
        results = {
            'job_id': args.job_id,
            'input_file': args.input,
            'n_particles': args.n_particles,
            'threads': args.threads,
            'seed': args.seed,
            'event_count': int(stats.counts.event_count),
            'track_count': int(stats.counts.track_count),
            'step_count': int(stats.counts.step_count),
        }
        
        # Guardar metadatos
        with open(output_folder / 'simulation_info.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📊 Estadísticas:")
        print(f"   Eventos:  {stats.counts.event_count:,}")
        print(f"   Tracks:   {stats.counts.track_count:,}")
        print(f"   Steps:    {stats.counts.step_count:,}")
        print(f"\n✅ Resultados guardados en: {output_folder}")
        
        return 0
        
    except Exception as e:
        print(f"⚠️  Advertencia en post-procesamiento: {e}")
        return 0  # No fallar si solo el post-procesamiento falla


def main():
    parser = argparse.ArgumentParser(
        description='Simulación OpenGate con PhaseSpaceSource (parametrizada para cluster)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Argumentos obligatorios
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Archivo ROOT con phase space (e.g., data/IAEA/phsp_500k.root)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Directorio de salida para resultados (se crea si no existe)'
    )
    
    # Parámetros de simulación
    parser.add_argument(
        '--n-particles', '-n',
        type=int,
        default=500000,
        help='Número de partículas a simular'
    )
    
    parser.add_argument(
        '--threads', '-t',
        type=int,
        default=1,
        help='Número de threads (1 para single-thread)'
    )
    
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=123,
        help='Semilla aleatoria para reproducibilidad'
    )
    
    # Opcionales
    parser.add_argument(
        '--job-id',
        type=str,
        default=None,
        help='ID del job en cluster (para tracking)'
    )
    
    parser.add_argument(
        '--compute-xy',
        action='store_true',
        help='Calcular también perfil transversal XY'
    )
    
    args = parser.parse_args()
    
    # Ejecutar
    exit_code = run_simulation(args)
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
