#!/usr/bin/env python3
"""
Training loop para U-Net 3D de denoising de dosis
Incluye: loss function, validation, checkpointing, logging
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
    """
    
    def __init__(self, model, device='cuda', learning_rate=1e-4, weight_decay=1e-5):
        self.model = model
        self.device = device
        
        # Optimizer
        self.optimizer = Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
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
        """Entrenar 1 epoch"""
        self.model.train()
        
        total_loss = 0
        num_batches = 0
        
        # Barra de progreso
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]", 
                   ncols=100, file=sys.stdout)
        
        for batch_idx, (noisy, clean) in enumerate(pbar):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            pred = self.model(noisy)
            loss = self.criterion(pred, clean)
            
            # Backward
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
        """Validación"""
        self.model.eval()
        
        total_loss = 0
        num_batches = 0
        
        # Barra de progreso para validación
        pbar = tqdm(val_loader, desc=f"[Val]  ", 
                   ncols=100, file=sys.stdout)
        
        for noisy, clean in pbar:
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)
            
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
        
        # Guardar el mejor
        if self.history['val_loss'][-1] < self.best_val_loss:
            best_path = checkpoint_dir / 'model_best.pth'
            torch.save(self.model.state_dict(), best_path)
            self.best_val_loss = self.history['val_loss'][-1]
        
        print(f"✅ Checkpoint saved: {checkpoint_path}")
    
    def save_history(self, history_file='training_history.json'):
        """Guardar historia de training"""
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"✅ History saved: {history_file}")
    
    def train(self, train_loader, val_loader, epochs=50, checkpoint_dir='checkpoints',
             early_stop_patience=15):
        """
        Loop de training completo
        
        Args:
            train_loader: DataLoader de training
            val_loader: DataLoader de validación
            epochs: Número de epochs
            checkpoint_dir: Directorio para checkpoints
            early_stop_patience: Paciencia para early stopping
        """
        print("="*60)
        print("🚀 STARTING TRAINING")
        print("="*60)
        print(f"Device: {self.device}")
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
    """Ejemplo de training"""
    
    # Configuración
    BATCH_SIZE = 4
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Device: {DEVICE}")
    
    # Crear dataloaders
    print("\n📂 Loading datasets...")
    train_loader, val_loader = get_dataloaders(
        batch_size=BATCH_SIZE,
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
    
    # Trainer
    print("\n🎓 Initializing trainer...")
    trainer = DoseDenoiser(
        model,
        device=DEVICE,
        learning_rate=LEARNING_RATE
    )
    
    # Training
    trainer.train(
        train_loader,
        val_loader,
        epochs=EPOCHS,
        checkpoint_dir='checkpoints',
        early_stop_patience=15
    )
    
    print("\n✅ Done!")


if __name__ == '__main__':
    main()
