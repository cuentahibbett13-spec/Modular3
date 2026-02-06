#!/usr/bin/env python3
"""
=============================================================
ENTRENAMIENTO CON VOLÚMENES COMPLETOS (300x150x150)
=============================================================
Usa volúmenes COMPLETOS sin parches para preservar contexto global.
Model: UNet3D base_ch=16 (más pequeño) + Mixed Precision.
"""

# ---- Desactivar MIOpen (ANTES de importar torch) ----
import os
os.environ["MIOPEN_DEBUG_DISABLE_FIND_DB"] = "1"
torch_backends_cudnn_enabled = False
# -----------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import SimpleITK as sitk

# Desactivar MIOpen después de importar torch
torch.backends.cudnn.enabled = False

# =============================================================
# ⚙️ VARIABLES DE CONFIGURACIÓN - REVISAR ANTES DE EJECUTAR
# =============================================================

# Rutas
DATASET_ROOT   = Path("dataset_pilot")          # Carpeta raíz del dataset
TRAIN_DIR      = DATASET_ROOT / "train"          # Subcarpeta de entrenamiento
VAL_DIR        = DATASET_ROOT / "val"            # Subcarpeta de validación
OUTPUT_DIR     = Path("runs/denoising_fullvol")  # Donde se guardan los checkpoints

# Niveles de input (carpetas o archivos dentro de cada pair)
INPUT_LEVELS   = ["input_1M", "input_2M", "input_5M", "input_10M"]

# Hiperparámetros
BATCH_SIZE     = 1                # Tamaño de batch (1 = volumen completo por paso)
NUM_EPOCHS     = 50               # Número de épocas
LEARNING_RATE  = 1e-3             # Learning rate
DEVICE         = "auto"           # "auto", "cuda" o "cpu"
USE_AMP        = True             # Mixed Precision para ahorrar memoria

# =============================================================
# FIN DE CONFIGURACIÓN
# =============================================================


def read_volume(mhd_path: Path) -> np.ndarray:
    """Lee un archivo .mhd y retorna un array 3D float32."""
    # Intentar .npy con el mismo nombre base
    npy_path = mhd_path.with_suffix(".npy")
    if npy_path.exists():
        return np.load(str(npy_path)).astype(np.float32)
    
    # Leer con SimpleITK
    img = sitk.ReadImage(str(mhd_path))
    arr = sitk.GetArrayFromImage(img)
    return arr.astype(np.float32)


class FullVolumeDataset(Dataset):
    """Dataset que devuelve volúmenes COMPLETOS sin parches."""
    
    def __init__(self, split_dir, dataset_root, levels, is_train=True):
        self.is_train = is_train
        self.pairs = []  # Lista de (input_path, target_path)
        
        n_targets = len(list(dataset_root.glob("target_*")))
        
        for pair_dir in sorted(split_dir.glob("pair_*")):
            pair_num = int(pair_dir.name.split("_")[-1])
            target_idx = ((pair_num - 1) % n_targets) + 1
            target_mhd = dataset_root / f"target_{target_idx}" / "dose_edep.mhd"
            
            if not target_mhd.exists():
                print(f"  ⚠️  Target no encontrado: {target_mhd}")
                continue
            
            for level in levels:
                # Buscar input: primero como archivo, luego como subdirectorio
                input_mhd = pair_dir / f"{level}.mhd"
                if not input_mhd.exists():
                    input_mhd = pair_dir / level / "dose_edep.mhd"
                
                if input_mhd.exists():
                    self.pairs.append((input_mhd, target_mhd))
                else:
                    print(f"  ⚠️  Input no encontrado: {pair_dir.name}/{level}")
        
        print(f"  ✅ {len(self.pairs)} pares encontrados en {split_dir.name}")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        input_path, target_path = self.pairs[idx]
        
        inp = read_volume(input_path)
        tgt = read_volume(target_path)
        
        # Normalizar por el máximo del target
        max_val = float(np.max(tgt))
        if max_val > 0:
            inp = inp / max_val
            tgt = tgt / max_val
        else:
            return None, None
        
        # A tensores con canal (SIN CROP - volumen completo)
        inp = torch.from_numpy(inp).unsqueeze(0)  # (1, Z, Y, X)
        tgt = torch.from_numpy(tgt).unsqueeze(0)
        
        return inp, tgt


# ---- Modelo UNet3D (base_ch=16 para ahorrar memoria) ----

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.GroupNorm(8, out_ch),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.GroupNorm(8, out_ch),
        nn.ReLU(inplace=True),
    )

class UNet3D(nn.Module):
    def __init__(self, base_ch=16):
        super().__init__()
        self.enc1 = conv_block(1, base_ch)
        self.enc2 = conv_block(base_ch, base_ch*2)
        self.enc3 = conv_block(base_ch*2, base_ch*4)
        self.pool = nn.MaxPool3d(2)
        self.bottleneck = conv_block(base_ch*4, base_ch*8)
        self.up3 = nn.ConvTranspose3d(base_ch*8, base_ch*4, 2, stride=2)
        self.dec3 = conv_block(base_ch*8, base_ch*4)
        self.up2 = nn.ConvTranspose3d(base_ch*4, base_ch*2, 2, stride=2)
        self.dec2 = conv_block(base_ch*4, base_ch*2)
        self.up1 = nn.ConvTranspose3d(base_ch*2, base_ch, 2, stride=2)
        self.dec1 = conv_block(base_ch*2, base_ch)
        self.out = nn.Conv3d(base_ch, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.out(d1)


# ---- Entrenamiento ----

def main():
    # Device
    if DEVICE == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(DEVICE)
    
    print("=" * 60)
    print("🧠 ENTRENAMIENTO CON VOLÚMENES COMPLETOS")
    print("=" * 60)
    print(f"📂 Dataset:     {DATASET_ROOT}")
    print(f"📂 Train dir:   {TRAIN_DIR}")
    print(f"📂 Val dir:     {VAL_DIR}")
    print(f"📂 Output:      {OUTPUT_DIR}")
    print(f"🔧 Device:      {device}")
    if torch.cuda.is_available():
        print(f"🔧 GPU:         {torch.cuda.get_device_name(0)}")
    print(f"📊 Batch size:  {BATCH_SIZE}")
    print(f"📊 Epochs:      {NUM_EPOCHS}")
    print(f"📊 LR:          {LEARNING_RATE}")
    print(f"📊 AMP:         {'Enabled' if USE_AMP else 'Disabled'}")
    print("=" * 60)
    
    # Verificar que existen las carpetas
    assert TRAIN_DIR.exists(), f"❌ No existe: {TRAIN_DIR}"
    assert VAL_DIR.exists(), f"❌ No existe: {VAL_DIR}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Datasets
    print("\n📂 Cargando datasets...")
    train_ds = FullVolumeDataset(TRAIN_DIR, DATASET_ROOT, INPUT_LEVELS, is_train=True)
    val_ds   = FullVolumeDataset(VAL_DIR, DATASET_ROOT, INPUT_LEVELS, is_train=False)
    
    assert len(train_ds) > 0, "❌ No hay datos de entrenamiento"
    assert len(val_ds) > 0, "❌ No hay datos de validación"
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    # Modelo
    model = UNet3D(base_ch=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 Modelo: {total_params:,} parámetros (base_ch=16)")
    
    # AMP Scaler
    scaler = torch.cuda.amp.GradScaler() if USE_AMP else None
    
    # Training loop
    best_val_loss = float("inf")
    
    print("\n🚀 Iniciando entrenamiento...\n")
    
    for epoch in range(1, NUM_EPOCHS + 1):
        # ---- Train ----
        model.train()
        train_losses = []
        
        for inp, tgt in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [train]", leave=False):
            if inp is None:
                continue
            
            inp = inp.to(device)
            tgt = tgt.to(device)
            
            optimizer.zero_grad()
            
            if USE_AMP:
                with torch.cuda.amp.autocast():
                    pred = model(inp)
                    loss = loss_fn(pred, tgt)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(inp)
                loss = loss_fn(pred, tgt)
                loss.backward()
                optimizer.step()
            
            train_losses.append(loss.item())
        
        train_loss = np.mean(train_losses) if train_losses else 0
        
        # ---- Validation ----
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for inp, tgt in val_loader:
                if inp is None:
                    continue
                inp = inp.to(device)
                tgt = tgt.to(device)
                
                if USE_AMP:
                    with torch.cuda.amp.autocast():
                        pred = model(inp)
                        loss = loss_fn(pred, tgt)
                else:
                    pred = model(inp)
                    loss = loss_fn(pred, tgt)
                
                val_losses.append(loss.item())
        
        val_loss = np.mean(val_losses) if val_losses else 0
        
        # ---- Log ----
        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss},
                       str(OUTPUT_DIR / "best.pt"))
            marker = " ⭐ BEST"
        
        print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | train_loss: {train_loss:.6f} | val_loss: {val_loss:.6f}{marker}")
        
        # Guardar checkpoint cada 10 épocas
        if epoch % 10 == 0:
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss},
                       str(OUTPUT_DIR / f"ckpt_epoch_{epoch:03d}.pt"))
    
    print(f"\n✅ Entrenamiento completado!")
    print(f"📊 Mejor val_loss: {best_val_loss:.6f}")
    print(f"📁 Checkpoints en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
