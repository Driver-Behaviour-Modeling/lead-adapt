"""
K-disks Clustering Algorithm for LEAD Motion Vocabulary.

Creates a vocabulary of motion primitives by:
1. Randomly selecting a sample from the data
2. Removing all samples within a tolerance distance of the selected sample
3. Repeating until the vocabulary is full or no disk is large enough

Adapted for LEAD CARLA expert trajectories.
"""

import os
import pickle

import numpy as np


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    """
    Wrap angle to [-π, π] range.

    Args:
        angle: Angle array in radians

    Returns:
        Wrapped angle in [-π, π]
    """
    return np.arctan2(np.sin(angle), np.cos(angle))


# Half-diagonal of the CARLA ego box (config_base.ego_extent_x/y = 2.451 m x 1.064 m
# half-extents), i.e. how far 1 rad of yaw displaces the vehicle's corners. This is
# the physically meaningful radians -> metres conversion for the K-disks metric.
EGO_HALF_DIAGONAL_M = float(np.hypot(2.4508416652679443, 1.0641621351242065))


def delta_distance(
    delta1: np.ndarray,
    delta2: np.ndarray,
    heading_weight: float = EGO_HALF_DIAGONAL_M,
) -> np.ndarray:
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
    pos_dist = np.sqrt(np.sum(pos_diff**2, axis=-1))

    # Heading distance (wrapped)
    heading_diff = wrap_angle(delta1[..., 2] - delta2[..., 2])
    heading_dist = np.abs(heading_diff) * heading_weight

    return pos_dist + heading_dist


def kdisks_cluster_deltas(
    deltas: np.ndarray,
    num_clusters: int = 4096,
    tolerance: float = 0.05,
    heading_weight: float = EGO_HALF_DIAGONAL_M,
    max_attempts: int = 100000,
    min_cluster_size: int = 25,
    reserve_stationary: bool = True,
    dx_bounds: tuple[float, float] = (-0.5, 9.0),
    dy_bound: float = 1.0,
    dheading_bound: float = 0.5,
    seed: int | None = None,
) -> tuple[np.ndarray, dict]:
    """
    K-disks clustering for body-frame motion deltas.

    Deltas are expected in the ego frame at the start of each step (see
    compute_deltas in scripts/extract_lead_carla_deltas.py), so that one physical
    manoeuvre maps to one code regardless of the ego's absolute heading.

    Args:
        deltas: [N, 3] array of body-frame motion deltas (Δx, Δy, Δheading)
        num_clusters: Upper bound on vocabulary size. The cover terminates early
            when no disk still meets min_cluster_size, so the returned count is
            frequently smaller.
        tolerance: Distance threshold for cluster membership
        heading_weight: Weight for heading in distance computation
        max_attempts: Maximum attempts to find valid clusters
        min_cluster_size: A disk holding fewer members than this does not get a
            code; its samples snap to their nearest surviving primitive at encode
            time. A centroid fitted to a handful of samples is an outlier the
            model can never learn to emit.
        reserve_stationary: Reserve token 0 for "no motion", a large fraction of
            driving frames that would otherwise distort the cover.
        dx_bounds: Plausible forward displacement per step in metres. Defaults
            suit LEAD's 0.25 s steps (up to 36 m/s forward, a little reverse).
        dy_bound: Plausible |lateral displacement| per step in metres.
        dheading_bound: Plausible |Δheading| per step in radians.
        seed: Random seed for reproducibility

    Returns:
        centroids: [K, 3] array of cluster centers, K <= num_clusters
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
    discarded_samples = 0
    rejected_candidates = 0

    if reserve_stationary:
        stationary = np.zeros(3, dtype=np.float64)
        within_tol = delta_distance(remaining, stationary, heading_weight) <= tolerance
        centroids.append(stationary)
        cluster_sizes.append(int(np.sum(within_tol)))
        remaining = remaining[~within_tol]
        print(f"Reserved token 0 for stationary: {cluster_sizes[0]} samples")

    attempts = 0
    while (
        len(centroids) < num_clusters and len(remaining) > 0 and attempts < max_attempts
    ):
        attempts += 1

        # Randomly select a sample
        idx = np.random.randint(len(remaining))
        candidate = remaining[idx]

        # Skip physically implausible candidates (route resets, pose glitches).
        if not (dx_bounds[0] <= candidate[0] <= dx_bounds[1]):
            rejected_candidates += 1
            continue
        if np.abs(candidate[1]) > dy_bound or np.abs(candidate[2]) > dheading_bound:
            rejected_candidates += 1
            continue

        # Compute distance to all remaining samples
        distances = delta_distance(remaining, candidate, heading_weight)

        # Find samples within tolerance
        within_tol = distances <= tolerance
        cluster_size = int(np.sum(within_tol))

        if cluster_size < min_cluster_size:
            # Retire the samples without spending a code.
            discarded_samples += cluster_size
            remaining = remaining[~within_tol]
            continue

        # Use mean of cluster as centroid
        cluster_samples = remaining[within_tol]

        # Handle heading averaging properly (circular mean)
        mean_x = cluster_samples[:, 0].mean()
        mean_y = cluster_samples[:, 1].mean()
        mean_heading = np.arctan2(
            np.sin(cluster_samples[:, 2]).mean(),
            np.cos(cluster_samples[:, 2]).mean(),
        )
        centroid = np.array([mean_x, mean_y, mean_heading])

        centroids.append(centroid)
        cluster_sizes.append(cluster_size)

        # Remove clustered samples
        remaining = remaining[~within_tol]

        if len(centroids) % 500 == 0:
            print(
                f"Created {len(centroids)}/{num_clusters} clusters, "
                f"{len(remaining)} samples remaining",
            )

    centroids = np.array(centroids)
    cluster_sizes = np.array(cluster_sizes)

    # Deliberately NOT padding to num_clusters with single-sample centroids: that
    # manufactures exactly the untrainable codes min_cluster_size exists to avoid.
    if len(centroids) < num_clusters:
        print(
            f"Cover terminated at {len(centroids)} clusters (cap {num_clusters}); "
            f"{discarded_samples} samples were below min_cluster_size={min_cluster_size} "
            f"and map to their nearest surviving primitive.",
        )

    # Effective vocabulary size: a balanced codebook has perplexity == K.
    probs = cluster_sizes / max(cluster_sizes.sum(), 1)
    probs = probs[probs > 0]
    perplexity = float(np.exp(-(probs * np.log(probs)).sum()))

    info = {
        "num_clusters": len(centroids),
        "cluster_sizes": cluster_sizes,
        "perplexity": perplexity,
        "tolerance": tolerance,
        "heading_weight": heading_weight,
        "min_cluster_size": min_cluster_size,
        "reserve_stationary": reserve_stationary,
        "discarded_samples": discarded_samples,
        "rejected_candidates": rejected_candidates,
        "total_samples": len(deltas),
        "attempts": attempts,
        "frame": "body",
    }

    return centroids, info


def assign_to_clusters(
    deltas: np.ndarray,
    centroids: np.ndarray,
    heading_weight: float = EGO_HALF_DIAGONAL_M,
) -> np.ndarray:
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
    info: dict,
    config: dict | None = None,
):
    """
    Save K-disks vocabulary to file.

    Args:
        filepath: Output path (.pkl)
        centroids: Cluster centroids
        info: Clustering statistics
        config: Optional configuration used for clustering
    """
    data = {"centroids": centroids, "info": info, "config": config, "version": "1.0"}

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(data, f)

    print(f"Saved vocabulary with {len(centroids)} clusters to {filepath}")


def load_kdisks_vocabulary(filepath: str) -> tuple[np.ndarray, dict]:
    """
    Load K-disks vocabulary from file.

    Args:
        filepath: Input path (.pkl)

    Returns:
        centroids: Cluster centroids
        info: Clustering statistics
    """
    with open(filepath, "rb") as f:
        data = pickle.load(f)

    return data["centroids"], data["info"]
