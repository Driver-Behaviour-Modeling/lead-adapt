"""
K-disks Clustering Algorithm for LEAD Motion Vocabulary.

Creates a fixed-size vocabulary of motion primitives by:
1. Randomly selecting a sample from the data
2. Removing all samples within a tolerance distance of the selected sample
3. Repeating until the vocabulary is full

Adapted for LEAD CARLA expert trajectories.
"""

import numpy as np
from typing import Tuple, Dict, Optional
import pickle
import os


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    """
    Wrap angle to [-π, π] range.
    
    Args:
        angle: Angle array in radians
        
    Returns:
        Wrapped angle in [-π, π]
    """
    return np.arctan2(np.sin(angle), np.cos(angle))


def delta_distance(delta1: np.ndarray, delta2: np.ndarray, 
                   heading_weight: float = 1.0) -> np.ndarray:
    """
    Compute distance between motion deltas.
    
    For ego vehicle motion, we weight position and heading components.
    
    Args:
        delta1: [N, 3] or [3] array of (Δx, Δy, Δheading)
        delta2: [M, 3] or [3] array of (Δx, Δy, Δheading)
        heading_weight: Weight for heading difference (radians → equivalent meters)
        
    Returns:
        Distance between deltas
    """
    # Position distance (Euclidean)
    pos_diff = delta1[..., :2] - delta2[..., :2]
    pos_dist = np.sqrt(np.sum(pos_diff ** 2, axis=-1))
    
    # Heading distance (wrapped)
    heading_diff = wrap_angle(delta1[..., 2] - delta2[..., 2])
    heading_dist = np.abs(heading_diff) * heading_weight
    
    return pos_dist + heading_dist


def kdisks_cluster_deltas(
    deltas: np.ndarray,
    num_clusters: int = 4096,
    tolerance: float = 0.05,
    heading_weight: float = 1.0,
    max_attempts: int = 100000,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, Dict]:
    """
    K-disks clustering for motion deltas.
    
    Operates directly on 3D motion deltas (Δx, Δy, Δheading) for ego vehicle.
    
    Args:
        deltas: [N, 3] array of motion deltas (Δx, Δy, Δheading)
        num_clusters: Target vocabulary size (default 4096)
        tolerance: Distance threshold for cluster membership
        heading_weight: Weight for heading in distance computation
        max_attempts: Maximum attempts to find valid clusters
        seed: Random seed for reproducibility
        
    Returns:
        centroids: [num_clusters, 3] array of cluster centers
        info: Dictionary with clustering statistics
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Ensure deltas are float64 for numerical stability
    deltas = deltas.astype(np.float64)
    
    # Normalize heading to [-π, π]
    deltas[:, 2] = wrap_angle(deltas[:, 2])
    
    # Track remaining samples
    remaining = deltas.copy()
    centroids = []
    cluster_sizes = []
    
    attempts = 0
    while len(centroids) < num_clusters and len(remaining) > 0 and attempts < max_attempts:
        attempts += 1
        
        # Randomly select a sample
        idx = np.random.randint(len(remaining))
        candidate = remaining[idx]
        
        # Optional: Skip outliers (extreme motion values)
        # For ego vehicle, reasonable bounds: |Δx| < 10m, |Δy| < 5m per timestep
        if np.abs(candidate[0]) > 10 or np.abs(candidate[1]) > 5:
            continue
        
        # Compute distance to all remaining samples
        distances = delta_distance(remaining, candidate, heading_weight)
        
        # Find samples within tolerance
        within_tol = distances <= tolerance
        cluster_size = np.sum(within_tol)
        
        # Use mean of cluster as centroid
        cluster_samples = remaining[within_tol]
        
        # Handle heading averaging properly (circular mean)
        mean_x = cluster_samples[:, 0].mean()
        mean_y = cluster_samples[:, 1].mean()
        mean_heading = np.arctan2(
            np.sin(cluster_samples[:, 2]).mean(),
            np.cos(cluster_samples[:, 2]).mean()
        )
        centroid = np.array([mean_x, mean_y, mean_heading])
        
        centroids.append(centroid)
        cluster_sizes.append(cluster_size)
        
        # Remove clustered samples
        remaining = remaining[~within_tol]
        
        if len(centroids) % 500 == 0:
            print(f"Created {len(centroids)}/{num_clusters} clusters, "
                  f"{len(remaining)} samples remaining")
    
    centroids = np.array(centroids)
    
    # If we couldn't create enough clusters, pad with remaining or warn
    if len(centroids) < num_clusters:
        print(f"Warning: Only created {len(centroids)} clusters "
              f"(target: {num_clusters})")
        if len(remaining) > 0:
            # Add remaining samples as single-element clusters
            needed = min(num_clusters - len(centroids), len(remaining))
            extra_indices = np.random.choice(len(remaining), needed, replace=False)
            extra_centroids = remaining[extra_indices]
            centroids = np.vstack([centroids, extra_centroids])
            cluster_sizes.extend([1] * needed)
    
    info = {
        'num_clusters': len(centroids),
        'cluster_sizes': np.array(cluster_sizes),
        'tolerance': tolerance,
        'heading_weight': heading_weight,
        'total_samples': len(deltas),
        'attempts': attempts
    }
    
    return centroids, info


def assign_to_clusters(deltas: np.ndarray, centroids: np.ndarray,
                       heading_weight: float = 1.0) -> np.ndarray:
    """
    Assign deltas to nearest cluster centroid.
    
    Args:
        deltas: [N, 3] array of motion deltas
        centroids: [K, 3] array of cluster centers
        heading_weight: Weight for heading in distance
        
    Returns:
        indices: [N] array of cluster indices
    """
    # Compute distances to all centroids
    # Shape: [N, K]
    distances = np.zeros((len(deltas), len(centroids)))
    
    for i, centroid in enumerate(centroids):
        distances[:, i] = delta_distance(deltas, centroid, heading_weight)
    
    return np.argmin(distances, axis=1)


def save_kdisks_vocabulary(
    filepath: str,
    centroids: np.ndarray,
    info: Dict,
    config: Optional[Dict] = None
):
    """
    Save K-disks vocabulary to file.
    
    Args:
        filepath: Output path (.pkl)
        centroids: Cluster centroids
        info: Clustering statistics
        config: Optional configuration used for clustering
    """
    data = {
        'centroids': centroids,
        'info': info,
        'config': config,
        'version': '1.0'
    }
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Saved vocabulary with {len(centroids)} clusters to {filepath}")


def load_kdisks_vocabulary(filepath: str) -> Tuple[np.ndarray, Dict]:
    """
    Load K-disks vocabulary from file.
    
    Args:
        filepath: Input path (.pkl)
        
    Returns:
        centroids: Cluster centroids
        info: Clustering statistics
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    return data['centroids'], data['info']
