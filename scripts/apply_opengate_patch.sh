#!/bin/bash
# 
# Script para aplicar el patch crítico de OpenGate en el cluster
# Este patch es necesario para que PhaseSpaceSource funcione con uproot 5.x
#
# IMPORTANTE: Ejecutar este script DESPUÉS de instalar OpenGate en el entorno
#
# Uso:
#   bash apply_opengate_patch.sh [path_to_venv]
#
# Ejemplo:
#   bash apply_opengate_patch.sh ~/.virtualenvs/modular3
#   bash apply_opengate_patch.sh .venv
#

set -e  # Exit on error

VENV_PATH="${1:-.venv}"
PYTHON_VERSION="python3.12"

echo "======================================================================"
echo "APLICANDO PATCH DE OPENGATE PARA UPROOT 5.x"
echo "======================================================================"
echo "Entorno: $VENV_PATH"
echo "======================================================================"

# Encontrar el archivo phspsources.py
PHSP_FILE="$VENV_PATH/lib/$PYTHON_VERSION/site-packages/opengate/sources/phspsources.py"

# Si no se encuentra con python3.12, intentar con python3.11
if [ ! -f "$PHSP_FILE" ]; then
    PYTHON_VERSION="python3.11"
    PHSP_FILE="$VENV_PATH/lib/$PYTHON_VERSION/site-packages/opengate/sources/phspsources.py"
fi

# Si aún no se encuentra, buscar dinámicamente
if [ ! -f "$PHSP_FILE" ]; then
    echo "⚠️  Buscando phspsources.py..."
    PHSP_FILE=$(find "$VENV_PATH" -name "phspsources.py" -path "*/opengate/sources/*" | head -n 1)
fi

if [ ! -f "$PHSP_FILE" ]; then
    echo "❌ ERROR: No se encuentra phspsources.py en $VENV_PATH"
    echo "   Verifica que OpenGate esté instalado correctamente"
    exit 1
fi

echo "✅ Encontrado: $PHSP_FILE"

# Crear backup
BACKUP="${PHSP_FILE}.backup_$(date +%Y%m%d_%H%M%S)"
cp "$PHSP_FILE" "$BACKUP"
echo "📦 Backup creado: $BACKUP"

# Verificar si ya está parcheado
if grep -q "If uproot returns a structured numpy array, convert to dict" "$PHSP_FILE"; then
    echo "✅ El archivo ya está parcheado. No se requiere acción."
    exit 0
fi

# Aplicar patch
echo "🔧 Aplicando patch..."

# Usar Python para hacer el patch de forma precisa
python3 << 'PYTHON_PATCH'
import sys
from pathlib import Path

phsp_file = Path(sys.argv[1])
content = phsp_file.read_text()

# Buscar la línea donde hacer el patch (después de self.batch = ...)
search_pattern = """        self.batch = self.root_file.arrays(
            entry_start=self.current_index,
            entry_stop=self.current_index + current_batch_size,
            library="numpy",
        )
        batch = self.batch"""

replacement = """        self.batch = self.root_file.arrays(
            entry_start=self.current_index,
            entry_stop=self.current_index + current_batch_size,
            library="numpy",
        )
        batch = self.batch
        # If uproot returns a structured numpy array, convert to dict of arrays
        if isinstance(batch, np.ndarray) and batch.dtype.names is not None:
            batch = {name: batch[name] for name in batch.dtype.names}
            self.batch = batch"""

if search_pattern in content:
    content = content.replace(search_pattern, replacement)
    phsp_file.write_text(content)
    print("✅ Patch aplicado correctamente en línea ~143")
    sys.exit(0)
else:
    print("⚠️  WARNING: No se encontró el patrón exacto. Puede que la versión de OpenGate sea diferente.")
    print("   Verifica manualmente el archivo:", phsp_file)
    sys.exit(1)

PYTHON_PATCH

# Pasar el archivo como argumento al script de Python
python3 -c "
import sys
from pathlib import Path

phsp_file = Path('$PHSP_FILE')
content = phsp_file.read_text()

search_pattern = '''        self.batch = self.root_file.arrays(
            entry_start=self.current_index,
            entry_stop=self.current_index + current_batch_size,
            library=\"numpy\",
        )
        batch = self.batch'''

replacement = '''        self.batch = self.root_file.arrays(
            entry_start=self.current_index,
            entry_stop=self.current_index + current_batch_size,
            library=\"numpy\",
        )
        batch = self.batch
        # If uproot returns a structured numpy array, convert to dict of arrays
        if isinstance(batch, np.ndarray) and batch.dtype.names is not None:
            batch = {name: batch[name] for name in batch.dtype.names}
            self.batch = batch'''

if search_pattern in content:
    content = content.replace(search_pattern, replacement)
    phsp_file.write_text(content)
    print('✅ Patch aplicado correctamente en línea ~143')
    sys.exit(0)
else:
    print('⚠️  WARNING: No se encontró el patrón exacto')
    print('   Archivo:', phsp_file)
    print('   Puede que la versión de OpenGate sea diferente')
    sys.exit(1)
"

PATCH_STATUS=$?

if [ $PATCH_STATUS -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "✅ PATCH APLICADO EXITOSAMENTE"
    echo "======================================================================"
    echo "El archivo phspsources.py ha sido modificado para soportar uproot 5.x"
    echo ""
    echo "CAMBIO REALIZADO:"
    echo "  - Agregadas 4 líneas después de 'batch = self.batch' (~línea 143)"
    echo "  - Convierte structured numpy arrays de uproot a dict"
    echo ""
    echo "VERIFICACIÓN:"
    echo "  grep -A 5 'batch = self.batch' '$PHSP_FILE'"
    echo ""
    echo "BACKUP:"
    echo "  $BACKUP"
    echo "======================================================================"
else
    echo ""
    echo "======================================================================"
    echo "⚠️  PATCH NO APLICADO AUTOMÁTICAMENTE"
    echo "======================================================================"
    echo "ACCIÓN REQUERIDA:"
    echo "  1. Editar manualmente: $PHSP_FILE"
    echo "  2. Buscar la línea ~143: 'batch = self.batch'"
    echo "  3. Agregar después:"
    echo ""
    echo "     # If uproot returns a structured numpy array, convert to dict"
    echo "     if isinstance(batch, np.ndarray) and batch.dtype.names is not None:"
    echo "         batch = {name: batch[name] for name in batch.dtype.names}"
    echo "         self.batch = batch"
    echo ""
    echo "BACKUP disponible en:"
    echo "  $BACKUP"
    echo "======================================================================"
    exit 1
fi
