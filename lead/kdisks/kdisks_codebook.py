"""
K-disks Codebook Module for LEAD.

Provides PyTorch-compatible interface for K-disks motion vocabulary.
Serves as a replacement for VQ-VAE in trajectory tokenization.

Key Features:
- Load pre-computed K-disks vocabulary
- Encode motion deltas to discrete tokens (nearest centroid)
- Decode tokens back to motion deltas (centroid lookup)
- GPU-accelerated distance computation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict
import pickle
from pathlib import Path


def wrap_angle_torch(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angle to [-π, π] range."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


class KDisksCodebook(nn.Module):
    """
    K-disks Codebook for motion tokenization.
    
    Unlike VQ-VAE which learns embeddings through training, K-disks uses
    a fixed codebook created by clustering. This module provides:
    - encode(): delta → token index (nearest centroid)
    - decode(): token index → delta (centroid value)
    
    Compatible with ADAPT's autoregressive decoder interface.
    """
    
    def __init__(
        self,
        vocab_path: Optional[str] = None,
        centroids: Optional[np.ndarray] = None,
        heading_weight: float = 1.0,
        normalize: bool = True
    ):
        """
        Initialize K-disks codebook.
        
        Args:
            vocab_path: Path to vocabulary file (.pkl)
            centroids: Alternatively, provide centroids directly [K, 3]
            heading_weight: Weight for heading in distance computation
            normalize: Whether to normalize deltas (for compatibility with VQ-VAE)
        """
        super().__init__()
        
        self.heading_weight = heading_weight
        self.normalize = normalize
        
        # Load or set centroids
        if vocab_path is not None:
            centroids, info = self._load_vocabulary(vocab_path)
            self._info = info
        elif centroids is not None:
            self._info = {'num_clusters': len(centroids)}
        else:
            raise ValueError("Either vocab_path or centroids must be provided")
        
        # Register centroids as buffer (not trainable)
        self.register_buffer('centroids', torch.from_numpy(centroids).float())
        
        # Compute normalization statistics from centroids
        if normalize:
            self.register_buffer('delta_mean', self.centroids.mean(dim=0))
            self.register_buffer('delta_std', self.centroids.std(dim=0) + 1e-8)
        
        self.num_embeddings = len(centroids)
        self.embedding_dim = 3  # (Δx, Δy, Δheading)
        
    def _load_vocabulary(self, path: str) -> Tuple[np.ndarray, Dict]:
        """Load K-disks vocabulary from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data['centroids'], data['info']
    
    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return self.num_embeddings
    
    def _compute_distances(
        self,
        deltas: torch.Tensor,
        centroids: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute distances between deltas and centroids.
        
        Args:
            deltas: [N, 3] tensor of motion deltas
            centroids: [K, 3] tensor of centroids
            
        Returns:
            distances: [N, K] tensor of distances
        """
        # Position distance
        pos_diff = deltas[:, :2].unsqueeze(1) - centroids[:, :2].unsqueeze(0)  # [N, K, 2]
        pos_dist = torch.norm(pos_diff, dim=-1)  # [N, K]
        
        # Heading distance (wrapped)
        heading_diff = deltas[:, 2:3] - centroids[:, 2:3].T  # [N, K]
        heading_diff = wrap_angle_torch(heading_diff)
        heading_dist = torch.abs(heading_diff) * self.heading_weight
        
        return pos_dist + heading_dist
    
    def encode(self, deltas: torch.Tensor) -> torch.Tensor:
        """
        Encode motion deltas to token indices (nearest centroid).
        
        Args:
            deltas: [N, 3] or [B, T, 3] tensor of motion deltas
            
        Returns:
            indices: [N, 1] or [B, T] tensor of token indices
        """
        # Handle batched input
        original_shape = deltas.shape
        if deltas.dim() == 3:
            B, T, D = deltas.shape
            deltas = deltas.reshape(-1, D)
        
        # Normalize heading
        deltas = deltas.clone()
        deltas[:, 2] = wrap_angle_torch(deltas[:, 2])
        
        # Compute distances to all centroids
        distances = self._compute_distances(deltas, self.centroids)  # [N, K]
        
        # Find nearest centroid
        indices = distances.argmin(dim=-1)  # [N]
        
        # Reshape to match expected output format
        if len(original_shape) == 3:
            indices = indices.reshape(B, T)
        else:
            indices = indices.unsqueeze(-1)  # [N, 1] for VQ-VAE compatibility
        
        return indices
    
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decode token indices to motion deltas (centroid lookup).
        
        Args:
            indices: [N] or [N, 1] or [B, T] tensor of token indices
            
        Returns:
            deltas: [N, 3] or [B, T, 3] tensor of motion deltas
        """
        # Handle various input shapes
        original_shape = indices.shape
        indices_flat = indices.view(-1)
        
        # Look up centroids
        decoded = self.centroids[indices_flat]  # [N, 3]
        
        # Reshape to match input
        if len(original_shape) == 2 and original_shape[-1] != 1:
            # [B, T] input
            B, T = original_shape
            decoded = decoded.reshape(B, T, 3)
        
        return decoded
    
    def quantize(self, deltas: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize deltas (for VQ-VAE interface compatibility).
        
        Args:
            deltas: [N, 3] tensor of motion deltas
            
        Returns:
            z_quantized: Quantized deltas (same as decoded centroids)
            dictionary_loss: Always 0 (no learned dictionary)
            commitment_loss: Quantization error
            encoding_indices: Token indices
        """
        # Encode to indices
        encoding_indices = self.encode(deltas)
        
        # Decode back to centroids
        z_quantized = self.decode(encoding_indices)
        if z_quantized.dim() == 3:
            z_quantized = z_quantized.squeeze(1)
        
        # Compute losses (for monitoring, not backprop)
        commitment_loss = torch.mean((deltas - z_quantized.detach()) ** 2)
        dictionary_loss = torch.tensor(0.0, device=deltas.device)  # No learned dictionary
        
        return z_quantized, dictionary_loss, commitment_loss, encoding_indices
    
    def forward(self, deltas: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for VQ-VAE interface compatibility.
        
        Args:
            deltas: [N, 3] tensor of motion deltas
            
        Returns:
            Dictionary with quantization outputs
        """
        z_quantized, dictionary_loss, commitment_loss, encoding_indices = self.quantize(deltas)
        
        return {
            'x_recon': z_quantized,
            'z_quantized': z_quantized,
            'encoding_indices': encoding_indices,
            'dictionary_loss': dictionary_loss,
            'commitment_loss': commitment_loss,
        }
    
    def get_embeddings(self) -> torch.Tensor:
        """
        Get codebook embeddings (centroids).
        
        Returns:
            embeddings: [K, 3] tensor of centroids
        """
        return self.centroids


class KDisksModel(nn.Module):
    """
    K-disks model wrapper that mimics VQReconModel interface.
    
    Provides a drop-in replacement for VQReconModel in ADAPT,
    maintaining the same API but using K-disks clustering instead of VQ-VAE.
    """
    
    def __init__(self, config):
        """
        Initialize K-disks model.
        
        Args:
            config: Configuration object with kdisks_vocab_path and other params
        """
        super().__init__()
        self._config = config
        
        # Initialize codebook
        self.codebook = KDisksCodebook(
            vocab_path=config.kdisks_vocab_path,
            heading_weight=getattr(config, 'kdisks_heading_weight', 1.0),
            normalize=getattr(config, 'normalize_kinematics', True)
        )
        
        # For compatibility with VQReconModel
        self.normalization_initialized = True
        
        # Store normalization buffers (for compatibility)
        if config.normalize_kinematics:
            self.register_buffer('delta_mean', self.codebook.delta_mean.clone())
            self.register_buffer('delta_std', self.codebook.delta_std.clone())
    
    def _compute_deltas(self, poses: torch.Tensor) -> torch.Tensor:
        """
        Compute frame-to-frame delta transformations.
        
        Args:
            poses: [batch_size, num_frames, 3] where 3 = (x, y, heading)
        
        Returns:
            deltas: [batch_size, num_frames-1, 3] where 3 = (Δx, Δy, Δheading)
        """
        delta_x = poses[:, 1:, 0] - poses[:, :-1, 0]
        delta_y = poses[:, 1:, 1] - poses[:, :-1, 1]
        delta_heading = poses[:, 1:, 2] - poses[:, :-1, 2]
        delta_heading = wrap_angle_torch(delta_heading)
        
        return torch.stack([delta_x, delta_y, delta_heading], dim=-1)
    
    def _normalize_angle(self, angle: torch.Tensor) -> torch.Tensor:
        """Normalize angle to [-π, π] range."""
        return wrap_angle_torch(angle)
    
    def encode(self, deltas: torch.Tensor) -> torch.Tensor:
        """
        Encode deltas to discrete codes.
        
        Args:
            deltas: [N, 3] or [B, T, 3] tensor of motion deltas
            
        Returns:
            encoding_indices: Token indices
        """
        return self.codebook.encode(deltas)
    
    def decode(self, encoding_indices: torch.Tensor) -> torch.Tensor:
        """
        Decode discrete codes to motion deltas.
        
        Args:
            encoding_indices: Token indices
            
        Returns:
            decoded: Motion deltas
        """
        return self.codebook.decode(encoding_indices)


def create_kdisks_vocabulary(
    deltas_path: str,
    output_path: str,
    num_clusters: int = 4096,
    tolerance: float = 0.05,
    heading_weight: float = 1.0,
    seed: int = 42
):
    """
    Create K-disks vocabulary from extracted deltas.
    
    Convenience function to run the full clustering pipeline.
    
    Args:
        deltas_path: Path to extracted deltas (.npy)
        output_path: Output path for vocabulary (.pkl)
        num_clusters: Target vocabulary size
        tolerance: Distance threshold for clustering
        heading_weight: Weight for heading in distance
        seed: Random seed
    """
    from .kdisks_clustering import kdisks_cluster_deltas, save_kdisks_vocabulary
    
    print(f"Loading deltas from {deltas_path}")
    deltas = np.load(deltas_path)
    print(f"Loaded {len(deltas)} deltas")
    
    print(f"Running K-disks clustering with {num_clusters} clusters...")
    centroids, info = kdisks_cluster_deltas(
        deltas,
        num_clusters=num_clusters,
        tolerance=tolerance,
        heading_weight=heading_weight,
        seed=seed
    )
    
    config = {
        'num_clusters': num_clusters,
        'tolerance': tolerance,
        'heading_weight': heading_weight,
        'seed': seed,
        'source': deltas_path
    }
    
    save_kdisks_vocabulary(output_path, centroids, info, config)
    print(f"Saved vocabulary to {output_path}")
    
    return centroids, info
