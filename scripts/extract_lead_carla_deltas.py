"""
Extract frame-to-frame motion deltas from LEAD CARLA expert trajectories.

Reads pickle files from CARLA expert data collection and extracts ego motion
deltas (Δx, Δy, Δheading) in local ego coordinates.

Usage:
    python lead/scripts/extract_lead_carla_deltas.py --verbose
"""

import argparse
import numpy as np
import pickle
import os
from pathlib import Path
import logging
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
LOG = logging.getLogger(__name__)


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    """Wrap angle to [-π, π]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def compute_deltas(poses: np.ndarray) -> np.ndarray:
    """
    Compute frame-to-frame deltas from poses.
    
    Args:
        poses: [N, 3] array of (x, y, heading) poses in local ego coordinates
        
    Returns:
        deltas: [N-1, 3] array of (Δx, Δy, Δheading) deltas
    """
    delta_x = poses[1:, 0] - poses[:-1, 0]
    delta_y = poses[1:, 1] - poses[:-1, 1]
    delta_heading = wrap_angle(poses[1:, 2] - poses[:-1, 2])
    
    return np.column_stack([delta_x, delta_y, delta_heading])


def transform_to_ego_frame(pose_global: np.ndarray, 
                           ego_pose_global: np.ndarray) -> np.ndarray:
    """
    Transform a global pose to ego-centric frame.
    
    Args:
        pose_global: [x, y, heading] in global coordinates
        ego_pose_global: [x, y, heading] ego pose in global coordinates
        
    Returns:
        pose_ego: [x, y, heading] relative to ego pose
    """
    # Extract components
    x, y, h = pose_global
    ego_x, ego_y, ego_h = ego_pose_global
    
    # Translate to ego origin
    dx = x - ego_x
    dy = y - ego_y
    
    # Rotate to ego frame (relative heading)
    cos_h = np.cos(ego_h)
    sin_h = np.sin(ego_h)
    x_ego = cos_h * dx + sin_h * dy
    y_ego = -sin_h * dx + cos_h * dy
    
    # Relative heading
    h_ego = wrap_angle(h - ego_h)
    
    return np.array([x_ego, y_ego, h_ego])


def extract_deltas_from_meta(meta_dict: dict) -> Optional[np.ndarray]:
    """
    Extract pose from a single meta dictionary from LEAD CARLA.
    
    Args:
        meta_dict: Pickle dict containing ego pose data
        
    Returns:
        pose: [x, y, heading] in local ego coordinates or None if invalid
    """
    try:
        # LEAD stores poses as [x, y] and theta (heading) separately
        pos_global = np.array(meta_dict.get("pos_global", [0, 0]))
        theta = meta_dict.get("theta", 0.0)
        
        # Return as local frame (already ego-centric after collection)
        return np.array([pos_global[0], pos_global[1], theta])
    except Exception as e:
        LOG.warning(f"Failed to extract pose from meta: {e}")
        return None


def extract_all_deltas_from_directory(
    carla_data_dir: Path,
    verbose: bool = True,
) -> np.ndarray:
    """
    Extract all deltas from a CARLA data directory.
    
    Scans for all route directories and collects deltas from each.
    
    Args:
        carla_data_dir: Root directory containing CARLA expert data
        verbose: Whether to print progress
        
    Returns:
        all_deltas: [N, 3] array of all extracted deltas
    """
    all_deltas = []
    total_trajectories = 0
    total_frames = 0
    
    # Find all metas directories recursively.
    # LEAD data may be nested as: root/scenario/route/metas
    if not carla_data_dir.exists():
        LOG.error(f"CARLA data directory not found: {carla_data_dir}")
        return np.array([]).reshape(0, 3)

    metas_dirs = sorted([d for d in carla_data_dir.rglob("metas") if d.is_dir()])
    if verbose:
        LOG.info(f"Found {len(metas_dirs)} metas directories")

    for metas_dir in metas_dirs:
        
        # Load all metas for this route
        poses = []
        meta_files = sorted(metas_dir.glob("*.pkl"))
        
        for meta_file in meta_files:
            try:
                with open(meta_file, 'rb') as f:
                    meta_dict = pickle.load(f)
                
                pose = extract_deltas_from_meta(meta_dict)
                if pose is not None:
                    poses.append(pose)
            except Exception as e:
                LOG.debug(f"Failed to load {meta_file}: {e}")
                continue
        
        # Compute deltas for this trajectory
        if len(poses) > 1:
            poses_array = np.array(poses)
            deltas = compute_deltas(poses_array)
            all_deltas.append(deltas)
            total_trajectories += 1
            total_frames += len(poses)
            
            if verbose and total_trajectories % 10 == 0:
                LOG.info(f"Processed {total_trajectories} trajectories, "
                        f"{len(np.vstack(all_deltas)) if all_deltas else 0} deltas")
    
    # Combine all deltas
    if all_deltas:
        all_deltas = np.vstack(all_deltas)
    else:
        all_deltas = np.array([]).reshape(0, 3)
    
    if verbose:
        LOG.info(f"Extracted {total_trajectories} trajectories with {total_frames} total frames")
        LOG.info(f"Total deltas: {len(all_deltas)}")
        
        if len(all_deltas) > 0:
            LOG.info(f"Delta statistics:")
            LOG.info(f"  Δx: mean={all_deltas[:, 0].mean():.4f}, "
                    f"std={all_deltas[:, 0].std():.4f}, "
                    f"range=[{all_deltas[:, 0].min():.4f}, {all_deltas[:, 0].max():.4f}]")
            LOG.info(f"  Δy: mean={all_deltas[:, 1].mean():.4f}, "
                    f"std={all_deltas[:, 1].std():.4f}, "
                    f"range=[{all_deltas[:, 1].min():.4f}, {all_deltas[:, 1].max():.4f}]")
            LOG.info(f"  Δh: mean={all_deltas[:, 2].mean():.4f}, "
                    f"std={all_deltas[:, 2].std():.4f}, "
                    f"range=[{all_deltas[:, 2].min():.4f}, {all_deltas[:, 2].max():.4f}]")
    
    return all_deltas


def main():
    parser = argparse.ArgumentParser(
        description="Extract CARLA expert motion deltas for k-disks vocabulary"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress information"
    )
    
    args = parser.parse_args()
    
    # Hardcoded paths based on LEAD structure
    lead_root = Path(__file__).parent.parent  # lead/scripts -> lead/
    adapt_dir = lead_root / "lead" / "adapt"
    data_dir = adapt_dir / "data"
    codebook_dir = adapt_dir / "codebooks"
    
    # Create output directories
    data_dir.mkdir(parents=True, exist_ok=True)
    codebook_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = data_dir / "carla_deltas.npy"
    
    # Determine CARLA data directory
    # First try environment variable, then use LEAD defaults
    carla_data_root = os.environ.get("LEAD_CARLA_DATA_ROOT", None)
    
    if carla_data_root is None:
        # Try common locations
        possible_paths = [
            lead_root / "data" / "carla_leaderboard2" / "data",
            lead_root / "data" / "data_routes",
            Path("/data/carla_expert"),
            Path("/mnt/data/carla_expert"),
        ]
        
        carla_data_root = None
        for path in possible_paths:
            if path.exists():
                carla_data_root = path
                break
    
    if carla_data_root is None:
        LOG.error("CARLA data directory not found. Set LEAD_CARLA_DATA_ROOT environment variable.")
        LOG.error("Expected paths checked:")
        for p in [
            str(lead_root / "data" / "carla_leaderboard2" / "data"),
            str(lead_root / "data" / "data_routes"),
            "/data/carla_expert",
            "/mnt/data/carla_expert",
        ]:
            LOG.error(f"  - {p}")
        return
    
    carla_data_root = Path(carla_data_root)
    
    if args.verbose:
        LOG.info(f"LEAD root: {lead_root}")
        LOG.info(f"Output directory: {data_dir}")
        LOG.info(f"CARLA data root: {carla_data_root}")
    
    # Extract deltas
    LOG.info("Extracting deltas from CARLA expert data...")
    deltas = extract_all_deltas_from_directory(carla_data_root, verbose=args.verbose)
    
    if len(deltas) == 0:
        LOG.error("No deltas extracted! Check CARLA data directory.")
        return
    
    # Save deltas
    np.save(str(output_path), deltas)
    LOG.info(f"Saved {len(deltas)} deltas to {output_path}")
    
    if args.verbose:
        LOG.info("Extraction complete!")
        LOG.info(f"Next step: Build vocabulary with:")
        LOG.info(f"  python lead/scripts/build_kdisks_carla.py --codebook_size 4096 --verbose")


if __name__ == "__main__":
    main()
