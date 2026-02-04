#!/usr/bin/env python3
"""
Training loop para U-Net 3D de denoising de dosis
Versión optimizada para AMD MI210 (ROCm) con 64GB VRAM
"""

import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from pathlib import Path
import json
import time
import sys
from datetime import datetime
from tqdm import tqdm

from dataset import get_dataloaders
from model import create_model

# Force unbuffered output
sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), 'w', buffering=1)


class DoseDenoiser:
    """
    Trainer para modelo U-Net 3D
    Optimizado para MI210 (batch_size 16, mixed precision)
    """
    
    def __init__(self, model, device='cuda', learning_rate=1e-4, weight_decay=1e-5, use_amp=True):
        self.model = model
        self.device = device
        self.use_amp = use_amp
        
        # Optimizer
        self.optimizer = Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Mixed precision training (FP16)
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Learning rate scheduler
        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
    
    def train_epoch(self, train_loader, epoch):
        """Entrenar 1 epoch con mixed precision"""
        self.model.train()
        
        total_loss = 0
        num_batches = 0
        
        # Barra de progreso
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]", 
                   ncols=100, file=sys.stdout)
        
        for batch_idx, (noisy, clean) in enumerate(pbar):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)
            
            # Forward con mixed precision
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    pred = self.model(noisy)
                    loss = self.criterion(pred, clean)
                
                # Backward con scaler
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred = self.model(noisy)
                loss = self.criterion(pred, clean)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_loss = total_loss / num_batches
        self.history['train_loss'].append(avg_loss)
        
        return avg_loss
    
    @torch.no_grad()
    def validate(self, val_loader):
        """Validación con mixed precision"""
        self.model.eval()
        
        total_loss = 0
        num_batches = 0
        
        # Barra de progreso para validación
        pbar = tqdm(val_loader, desc=f"[Val]  ", 
                   ncols=100, file=sys.stdout)
        
        for noisy, clean in pbar:
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    pred = self.model(noisy)
                    loss = self.criterion(pred, clean)
            else:
                pred = self.model(noisy)
                loss = self.criterion(pred, clean)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_loss = total_loss / num_batches
        self.history['val_loss'].append(avg_loss)
        
        return avg_loss
    
    def save_checkpoint(self, epoch, checkpoint_dir='checkpoints'):
        """Guardar checkpoint"""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f'model_epoch_{epoch:03d}.pth'
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"✅ Checkpoint saved: {checkpoint_path}")
    
    def save_best_model(self, val_loss, checkpoint_dir='checkpoints'):
        """Guardar mejor modelo"""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(exist_ok=True)
            
            best_path = checkpoint_dir / 'best_model.pth'
            torch.save(self.model.state_dict(), best_path)
    
    def save_history(self, history_path='training_history.json'):
        """Guardar historia de training"""
        history_path = Path(history_path)
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"📊 History saved: {history_path}")
    
    def train(self, train_loader, val_loader, epochs=100, early_stop_patience=10, checkpoint_dir='checkpoints'):
        """
        Training loop completo
        """
        print("="*60)
        print("🚀 STARTING TRAINING (MI210 Optimized)")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.use_amp}")
        print(f"Epochs: {epochs}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
        print("="*60)
        
        start_time = time.time()
        epoch_times = []
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Training
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_loss = self.validate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            self.history['learning_rate'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # Calcular tiempo de epoch
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)
            
            # Estimar tiempo restante
            if len(epoch_times) > 0:
                avg_epoch_time = sum(epoch_times) / len(epoch_times)
                remaining_epochs = epochs - (epoch + 1)
                eta_seconds = avg_epoch_time * remaining_epochs
                eta_hours = eta_seconds / 3600
                eta_str = f"{eta_hours:.1f}h" if eta_hours >= 1 else f"{eta_seconds/60:.0f}m"
            else:
                eta_str = "calculating..."
            
            # Print results
            print(f"\n📊 Epoch {epoch + 1}/{epochs} | Time: {epoch_time:.1f}s | ETA: {eta_str}")
            print(f"  Train loss: {train_loss:.6f}")
            print(f"  Val loss:   {val_loss:.6f}")
            print(f"  LR:         {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.early_stop_counter = 0
                print(f"  ✅ Best model! (improved from {self.best_val_loss:.6f})")
                self.save_best_model(val_loss, checkpoint_dir)
            else:
                self.early_stop_counter += 1
                print(f"  ⚠️  No improvement ({self.early_stop_counter}/{early_stop_patience})")
            
            # Checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch, checkpoint_dir)
            
            # Early stopping
            if self.early_stop_counter >= early_stop_patience:
                print(f"\n⛔ Early stopping after {epoch + 1} epochs")
                break
        
        # Resumen final
        elapsed = time.time() - start_time
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETED")
        print("="*60)
        print(f"Total time: {elapsed/3600:.1f} hours")
        print(f"Best val loss: {self.best_val_loss:.6f}")
        print(f"Final train loss: {train_loss:.6f}")
        print(f"Final val loss: {val_loss:.6f}")
        print("="*60)
        
        self.save_history()


def main():
    """Entrenamiento para MI210"""
    
    # Configuración optimizada para MI210 (64GB VRAM)
    BATCH_SIZE = 16  # vs 4 en RTX 5060
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    USE_AMP = True  # Mixed precision (FP16) para aprovechar mejor la VRAM
    
    print(f"Device: {DEVICE}")
    if DEVICE == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Crear dataloaders
    print("\n📂 Loading datasets...")
    train_loader, val_loader = get_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=4,  # vs 0 en GPU móvil
        augment=True,
        seed=42
    )
    
    # Crear modelo
    print("\n🏗️  Creating model...")
    model = create_model(
        device=DEVICE,
        base_channels=32,
        depth=4
    )
    
    # Crear trainer
    print("\n🎓 Initializing trainer...")
    trainer = DoseDenoiser(
        model=model,
        device=DEVICE,
        learning_rate=LEARNING_RATE,
        use_amp=USE_AMP
    )
    
    # Training
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        early_stop_patience=10,
        checkpoint_dir='checkpoints'
    )


if __name__ == '__main__':
    main()
