#!/usr/bin/env python3
"""
Convierte archivo IAEA phase space (.IAEAphsp) a formato NumPy (.npz).

FORMATO CORRECTO para Varian_Clinac_2100CD_6MeV (PENELOPE):
- Byte 0: particle type (signed char)
    |ptype| = 1: fotón, 2: electrón, 3: positrón
    signo negativo = partícula secundaria
- Bytes 1-28: 7 floats en orden:
    E_signed: energía en MeV (signo = dirección W)
    X: posición X en cm
    Y: posición Y en cm
    Z: posición Z en cm (~78.45, constante)
    U: coseno director en X
    V: coseno director en Y
    Weight: peso estadístico (típicamente 1)
- Bytes 29-36: 2 int32 (extra longs para PENELOPE)

W se calcula como: sign(E) * sqrt(1 - U² - V²)
"""

import numpy as np
from pathlib import Path
import argparse
import struct


def parse_iaea_header(header_file):
    """Lee el archivo .IAEAheader y extrae información relevante."""
    info = {}
    with open(header_file, 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    current_key = None
    
    for line in lines:
        line = line.strip()
        if line.startswith('$'):
            current_key = line[1:].rstrip(':')
        elif current_key and line and not line.startswith('//'):
            if current_key == 'RECORD_LENGTH':
                info['record_length'] = int(line)
            elif current_key == 'PARTICLES':
                info['n_particles'] = int(line)
            elif current_key == 'PHOTONS':
                info['n_photons'] = int(line)
            elif current_key == 'ELECTRONS':
                info['n_electrons'] = int(line)
            elif current_key == 'POSITRONS':
                info['n_positrons'] = int(line)
    
    return info


def read_iaea_phsp_corrected(phsp_file, header_info, max_particles=None):
    """
    Lee archivo binario IAEA phase space con el formato CORRECTO.
    
    Formato del registro (37 bytes) para Varian Clinac PENELOPE:
    - 1 byte: tipo de partícula (1=fotón, 2=electrón, 3=positrón)
    - 7 floats (28 bytes) en orden:
        - E_signed: energía en MeV, signo indica dirección de W
        - X: posición X en cm
        - Y: posición Y en cm
        - Z: posición Z en cm (constante ~78.45)
        - U: coseno director en X
        - V: coseno director en Y
        - Weight: peso estadístico
    - 2 int32 (8 bytes): extra longs
    """
    record_length = header_info.get('record_length', 37)
    n_particles = header_info.get('n_particles', 0)
    
    if max_particles:
        n_particles = min(n_particles, max_particles)
    
    print(f"Leyendo {n_particles:,} partículas...")
    print(f"Record length: {record_length} bytes")
    
    # Arrays para almacenar datos
    pos_x = np.zeros(n_particles, dtype=np.float32)
    pos_y = np.zeros(n_particles, dtype=np.float32)
    pos_z = np.zeros(n_particles, dtype=np.float32)
    dir_u = np.zeros(n_particles, dtype=np.float32)
    dir_v = np.zeros(n_particles, dtype=np.float32)
    dir_w = np.zeros(n_particles, dtype=np.float32)
    energy = np.zeros(n_particles, dtype=np.float32)
    weight = np.zeros(n_particles, dtype=np.float32)
    pdg = np.zeros(n_particles, dtype=np.int32)
    
    # Formato: '<' = little endian, 'b' = int8, 'f' = float32, 'i' = int32
    record_format = '<b7f2i'  # 1 + 28 + 8 = 37 bytes
    
    # Mapeo de tipo PENELOPE a PDG
    ptype_to_pdg = {1: 22, 2: 11, 3: -11}  # fotón, electrón, positrón
    
    with open(phsp_file, 'rb') as f:
        for i in range(n_particles):
            if i % 5_000_000 == 0 and i > 0:
                print(f"  Procesadas {i:,} partículas...")
            
            data = f.read(record_length)
            if len(data) < record_length:
                print(f"  Fin de archivo en partícula {i}")
                n_particles = i
                break
            
            try:
                values = struct.unpack(record_format, data)
                
                # ORDEN CORRECTO de los campos:
                ptype = values[0]      # tipo de partícula
                e_signed = values[1]   # energía con signo
                x = values[2]          # X en cm
                y = values[3]          # Y en cm
                z = values[4]          # Z en cm
                u = values[5]          # cos director X
                v = values[6]          # cos director Y
                wt = values[7]         # peso
                
                # Energía y dirección W
                e = abs(e_signed)
                w_sign = 1 if e_signed > 0 else -1
                w = w_sign * np.sqrt(max(0, 1 - u**2 - v**2))
                
                pos_x[i] = x
                pos_y[i] = y
                pos_z[i] = z
                dir_u[i] = u
                dir_v[i] = v
                dir_w[i] = w
                energy[i] = e
                weight[i] = wt
                
                # Tipo de partícula (usar valor absoluto)
                pt = abs(ptype)
                pdg[i] = ptype_to_pdg.get(pt, 0)
                    
            except struct.error as e:
                print(f"  Error en partícula {i}: {e}")
                continue
    
    # Truncar arrays si no leímos todas las partículas
    return {
        'pos_x': pos_x[:n_particles],
        'pos_y': pos_y[:n_particles],
        'pos_z': pos_z[:n_particles],
        'dir_u': dir_u[:n_particles],
        'dir_v': dir_v[:n_particles],
        'dir_w': dir_w[:n_particles],
        'energy': energy[:n_particles],
        'weight': weight[:n_particles],
        'pdg': pdg[:n_particles]
    }


def main():
    parser = argparse.ArgumentParser(
        description='Convierte IAEA phase space a NumPy (formato corregido)'
    )
    parser.add_argument('--input', required=True, help='Archivo .IAEAphsp de entrada')
    parser.add_argument('--output', required=True, help='Archivo .npz de salida')
    parser.add_argument('--max-particles', type=int, help='Límite de partículas')
    
    args = parser.parse_args()
    
    phsp_file = Path(args.input)
    header_file = phsp_file.with_suffix('.IAEAheader')
    
    if not header_file.exists():
        print(f"Error: No se encontró header {header_file}")
        return 1
    
    print(f"Leyendo header: {header_file}")
    header_info = parse_iaea_header(header_file)
    print(f"  Partículas totales: {header_info.get('n_particles', 'N/A'):,}")
    print(f"  Electrones: {header_info.get('n_electrons', 'N/A'):,}")
    print(f"  Fotones: {header_info.get('n_photons', 'N/A'):,}")
    
    print(f"\nLeyendo phase space: {phsp_file}")
    data = read_iaea_phsp_corrected(phsp_file, header_info, args.max_particles)
    
    # Estadísticas de verificación
    print("\n=== ESTADÍSTICAS DE VERIFICACIÓN ===")
    n_total = len(data['energy'])
    
    for pdg_code, name, expected_e in [(22, 'Fotón', 0.84), (11, 'Electrón', 6.13), (-11, 'Positrón', 1.60)]:
        mask = data['pdg'] == pdg_code
        n = mask.sum()
        if n > 0:
            e_mean = data['energy'][mask].mean()
            e_std = data['energy'][mask].std()
            print(f"  {name}: N={n:,} ({100*n/n_total:.1f}%), <E>={e_mean:.4f}±{e_std:.4f} MeV (esperado ~{expected_e})")
    
    # Guardar NPZ
    print(f"\nGuardando: {args.output}")
    np.savez_compressed(
        args.output,
        pos_x=data['pos_x'],
        pos_y=data['pos_y'],
        pos_z=data['pos_z'],
        dir_u=data['dir_u'],
        dir_v=data['dir_v'],
        dir_w=data['dir_w'],
        energy=data['energy'],
        weight=data['weight'],
        pdg=data['pdg']
    )
    
    print(f"✅ Conversión completada: {n_total:,} partículas")
    return 0


if __name__ == '__main__':
    exit(main())
