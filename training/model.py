#!/usr/bin/env python3
"""
U-Net 3D para denoising de dosis
Arquitectura optimizada para 125³ voxels con 8GB VRAM
"""

import torch
import torch.nn as nn
from typing import List


class ConvBlock3D(nn.Module):
    """Bloque convolutivo 3D: Conv → BN → ReLU → Conv → BN → ReLU"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class UNet3D(nn.Module):
    """
    U-Net 3D para denoising
    
    Arquitectura:
    - Input: [B, 1, 125, 125, 125]
    - Encoder: 4 niveles (con max pooling)
    - Decoder: 4 niveles (con upsampling)
    - Skip connections
    - Output: [B, 1, 125, 125, 125]
    
    Parámetros:
    - Base: 32 canales
    - Profundidad: 4 niveles
    - Memory-efficient: ~40M parámetros, 200MB en FP32
    """
    
    def __init__(self, in_channels=1, out_channels=1, base_channels=32, depth=4):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.depth = depth
        
        # =====================================================================
        # ENCODER
        # =====================================================================
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool3d(2)
        
        in_ch = in_channels
        for i in range(depth):
            out_ch = base_channels * (2 ** i)
            self.encoder.append(ConvBlock3D(in_ch, out_ch))
            in_ch = out_ch
        
        # =====================================================================
        # BOTTLENECK (nivel más profundo)
        # =====================================================================
        self.bottleneck = ConvBlock3D(
            base_channels * (2 ** (depth - 1)),
            base_channels * (2 ** depth)
        )
        
        # =====================================================================
        # DECODER
        # =====================================================================
        self.decoder = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        
        for i in range(depth - 1, -1, -1):
            # Upsample desde el nivel más profundo
            in_ch_up = base_channels * (2 ** (i + 1))
            out_ch_up = base_channels * (2 ** i)
            
            self.upsamples.append(
                nn.ConvTranspose3d(in_ch_up, out_ch_up, 2, stride=2)
            )
            
            # Conv después de concatenar skip connection
            # [out_ch_up (upsampled) + out_ch_up (skip)] = 2*out_ch_up
            self.decoder.append(
                ConvBlock3D(out_ch_up * 2, out_ch_up)
            )
        
        # =====================================================================
        # OUTPUT
        # =====================================================================
        self.final = nn.Conv3d(base_channels, out_channels, 1)
    
    def forward(self, x):
        """
        Args:
            x: [B, 1, 125, 125, 125]
        
        Returns:
            out: [B, 1, 125, 125, 125]
        """
        # =====================================================================
        # ENCODING PASS
        # =====================================================================
        skip_connections = []
        
        for i, conv_block in enumerate(self.encoder):
            x = conv_block(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # =====================================================================
        # BOTTLENECK
        # =====================================================================
        x = self.bottleneck(x)
        
        # =====================================================================
        # DECODING PASS
        # =====================================================================
        skip_connections = skip_connections[::-1]  # Invertir
        
        for i, (upsample, conv_block) in enumerate(
            zip(self.upsamples, self.decoder)
        ):
            # Upsample
            x = upsample(x)
            
            # Concatenar skip connection con match de tamaño
            skip = skip_connections[i]
            
            # Ajustar tamaño si no coincide
            if x.shape != skip.shape:
                # Pad o crop
                diff_d = skip.shape[2] - x.shape[2]
                diff_h = skip.shape[3] - x.shape[3]
                diff_w = skip.shape[4] - x.shape[4]
                
                x = nn.functional.pad(x, [
                    diff_w // 2, diff_w - diff_w // 2,
                    diff_h // 2, diff_h - diff_h // 2,
                    diff_d // 2, diff_d - diff_d // 2
                ])
            
            x = torch.cat([x, skip], dim=1)
            
            # Convolución
            x = conv_block(x)
        
        # =====================================================================
        # OUTPUT
        # =====================================================================
        x = self.final(x)
        
        return x
    
    def count_parameters(self):
        """Contar parámetros totales"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def print_summary(self):
        """Imprimir resumen del modelo"""
        total_params = self.count_parameters()
        
        print("="*60)
        print("📊 U-NET 3D SUMMARY")
        print("="*60)
        print(f"Total parameters: {total_params:,}")
        print(f"Model size (FP32): {total_params * 4 / 1024 / 1024:.1f} MB")
        print(f"Model size (FP16): {total_params * 2 / 1024 / 1024:.1f} MB")
        print(f"\nArchitecture:")
        print(f"  Input channels: {self.in_channels}")
        print(f"  Output channels: {self.out_channels}")
        print(f"  Base channels: {self.base_channels}")
        print(f"  Depth: {self.depth} levels")
        print(f"  Input size: [B, 1, 125, 125, 125]")
        print(f"  Output size: [B, 1, 125, 125, 125]")
        print("="*60)


def create_model(device='cuda', base_channels=32, depth=4, pretrained=None):
    """
    Crear modelo U-Net 3D
    
    Args:
        device: 'cuda' o 'cpu'
        base_channels: Canales base (32 por defecto)
        depth: Profundidad (4 por defecto)
        pretrained: Ruta a checkpoint preentrenado (opcional)
    
    Returns:
        model: U-Net3D en el device especificado
    """
    model = UNet3D(
        in_channels=1,
        out_channels=1,
        base_channels=base_channels,
        depth=depth
    )
    
    model = model.to(device)
    
    if pretrained:
        print(f"Loading pretrained weights from {pretrained}...")
        checkpoint = torch.load(pretrained, map_location=device)
        model.load_state_dict(checkpoint)
    
    model.print_summary()
    
    return model


if __name__ == '__main__':
    # Test
    print("Testing U-Net 3D...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    # Crear modelo
    model = create_model(device=device, base_channels=32, depth=4)
    
    # Test forward pass
    print("\nTesting forward pass...")
    x = torch.randn(2, 1, 125, 125, 125).to(device)
    
    with torch.no_grad():
        y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output dtype: {y.dtype}")
    
    # Estimar VRAM
    estimated_vram_mb = model.count_parameters() * 4 / 1024 / 1024 * 3  # 3x para gradientes
    print(f"\nEstimated training VRAM (batch=2): ~{estimated_vram_mb:.0f} MB")
    
    print("\n✅ Model funcionando correctamente")
