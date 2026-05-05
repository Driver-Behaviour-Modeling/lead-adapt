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

try:
    import zstandard as zstd
except Exception:  # pragma: no cover - optional dependency
    zstd = None

import lead.common.common_utils as common_utils

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
LOG = logging.getLogger(__name__)


def load_meta_file(meta_path: Path) -> dict:
    """Load a meta file that may be raw pickle or zstd-compressed pickle."""
    with open(meta_path, "rb") as f:
        header = f.read(4)
        f.seek(0)
        # Zstandard magic bytes: 28 B5 2F FD (little endian read often starts with FD)
        if header in (b"\x28\xb5\x2f\xfd", b"\xfd\x2f\xb5\x28"):
            if zstd is None:
                raise RuntimeError(
                    "zstandard is required to read compressed meta files. "
                    "Install with: pip install zstandard"
                )
            dctx = zstd.ZstdDecompressor()
            decompressed = dctx.decompress(f.read())
            return pickle.loads(decompressed)
        return pickle.load(f)


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    """Wrap angle to [-π, π]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def compute_deltas(poses: np.ndarray) -> np.ndarray:
    """
    Compute frame-to-frame deltas from poses.
    
    Args:
        poses: [N, 3] array of (x, y, heading) poses in global coordinates
        
    Returns:
        deltas: [N-1, 3] array of (Δx, Δy, Δheading) deltas in the local frame of the previous step
    """
    if len(poses) < 2:
        raise ValueError(f"Need at least 2 poses to compute deltas, got {len(poses)}")
    
    delta_x_global = poses[1:, 0] - poses[:-1, 0]
    delta_y_global = poses[1:, 1] - poses[:-1, 1]
    delta_heading = wrap_angle(poses[1:, 2] - poses[:-1, 2])
    
    # Transform position deltas from global to ego frame using common_utils
    # (same rotation used by Plant agent for route transformation)
    headings = poses[:-1, 2]
    delta_xy_global = np.column_stack([delta_x_global, delta_y_global])
    
    try:
        # Apply inverse_conversion_2d with zero translation (only rotation matters for deltas)
        delta_xy_local = np.array([
            common_utils.inverse_conversion_2d(
                point=delta_xy_global[i],
                translation=np.array([0.0, 0.0]),  # No translation for deltas
                yaw=headings[i]
            )
            for i in range(len(headings))
        ])
    except Exception as e:
        LOG.error(f"Error in inverse_conversion_2d: {e}")
        LOG.error(f"  delta_xy_global[0]: {delta_xy_global[0]}")
        LOG.error(f"  headings[0]: {headings[0]}")
        raise
    
    return np.column_stack([delta_xy_local[:, 0], delta_xy_local[:, 1], delta_heading])



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
        if "pos_global" not in meta_dict:
            LOG.debug(f"Missing 'pos_global' in meta dict, keys: {meta_dict.keys()}")
            return None
        
        if "theta" not in meta_dict:
            LOG.debug(f"Missing 'theta' in meta dict, keys: {meta_dict.keys()}")
            return None
        
        pos_global = np.array(meta_dict.get("pos_global", [0, 0]))
        theta = meta_dict.get("theta", 0.0)
        
        # Validate pose data
        if pos_global.shape != (2,):
            LOG.debug(f"Invalid pos_global shape: {pos_global.shape}, expected (2,)")
            return None
        
        if not isinstance(theta, (int, float, np.number)):
            LOG.debug(f"Invalid theta type: {type(theta)}")
            return None
        
        return np.array([pos_global[0], pos_global[1], theta], dtype=np.float32)
    except Exception as e:
        LOG.debug(f"Failed to extract pose from meta: {e}")
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
    skipped_short_trajectories = 0
    failed_extractions = 0
    
    # Find all metas directories recursively.
    # LEAD data may be nested as: root/scenario/route/metas
    if not carla_data_dir.exists():
        LOG.error(f"CARLA data directory not found: {carla_data_dir}")
        return np.array([]).reshape(0, 3)

    metas_dirs = sorted([d for d in carla_data_dir.rglob("metas") if d.is_dir()])
    if verbose:
        LOG.info(f"Found {len(metas_dirs)} metas directories")
        LOG.info(f"Starting extraction...")

    for idx, metas_dir in enumerate(metas_dirs):
        # Progress indicator
        if verbose and (idx + 1) % 100 == 0:
            LOG.info(f"  [{idx + 1}/{len(metas_dirs)}] Processing metas directory...")
        elif verbose and (idx + 1) % 500 == 0:
            LOG.info(f"  [{idx + 1}/{len(metas_dirs)}] {total_trajectories} trajectories extracted so far, "
                    f"{len(np.vstack(all_deltas)) if all_deltas else 0} deltas")
        
        # Load all metas for this route
        poses = []
        meta_files = sorted(metas_dir.glob("*.pkl"))
        
        if verbose and len(meta_files) == 0:
            LOG.debug(f"  {metas_dir.relative_to(carla_data_dir)}: No pickle files found")
        
        extraction_failed_count = 0
        for meta_file in meta_files:
            try:
                meta_dict = load_meta_file(meta_file)
                
                pose = extract_deltas_from_meta(meta_dict)
                if pose is not None:
                    poses.append(pose)
                else:
                    extraction_failed_count += 1
            except Exception as e:
                extraction_failed_count += 1
                LOG.debug(f"  Failed to load {meta_file.name}: {e}")
                continue
        
        if extraction_failed_count > 0 and verbose:
            LOG.debug(f"  {metas_dir.relative_to(carla_data_dir)}: "
                     f"{extraction_failed_count}/{len(meta_files)} meta files failed to extract")
        
        # Compute deltas for this trajectory
        if len(poses) > 1:
            try:
                poses_array = np.array(poses)
                deltas = compute_deltas(poses_array)
                all_deltas.append(deltas)
                total_trajectories += 1
                total_frames += len(poses)
                
                if verbose and (total_trajectories % 100 == 0):
                    LOG.info(f"  ✓ {total_trajectories} trajectories, "
                            f"{len(np.vstack(all_deltas))} total deltas extracted")
            except Exception as e:
                failed_extractions += 1
                LOG.warning(f"  Failed to compute deltas for {metas_dir.relative_to(carla_data_dir)}: {e}")
        elif len(poses) == 1:
            skipped_short_trajectories += 1
            if verbose and (skipped_short_trajectories % 100 == 0):
                LOG.debug(f"  Skipped {skipped_short_trajectories} trajectories with only 1 frame")
        elif len(poses) == 0 and extraction_failed_count > 0:
            failed_extractions += 1
    
    # Combine all deltas
    if all_deltas:
        all_deltas = np.vstack(all_deltas)
    else:
        all_deltas = np.array([]).reshape(0, 3)
    
    if verbose:
        LOG.info("")
        LOG.info("=" * 80)
        LOG.info("EXTRACTION SUMMARY")
        LOG.info("=" * 80)
        LOG.info(f"Metas directories processed: {len(metas_dirs)}")
        LOG.info(f"Successful trajectories: {total_trajectories}")
        LOG.info(f"Total frames: {total_frames}")
        LOG.info(f"Total deltas extracted: {len(all_deltas)}")
        LOG.info(f"Skipped (single-frame): {skipped_short_trajectories}")
        LOG.info(f"Failed extractions: {failed_extractions}")
        
        if len(all_deltas) > 0:
            LOG.info("")
            LOG.info("Delta statistics:")
            LOG.info(f"  Δx: mean={all_deltas[:, 0].mean():.4f}, "
                    f"std={all_deltas[:, 0].std():.4f}, "
                    f"range=[{all_deltas[:, 0].min():.4f}, {all_deltas[:, 0].max():.4f}]")
            LOG.info(f"  Δy: mean={all_deltas[:, 1].mean():.4f}, "
                    f"std={all_deltas[:, 1].std():.4f}, "
                    f"range=[{all_deltas[:, 1].min():.4f}, {all_deltas[:, 1].max():.4f}]")
            LOG.info(f"  Δh: mean={all_deltas[:, 2].mean():.4f}, "
                    f"std={all_deltas[:, 2].std():.4f}, "
                    f"range=[{all_deltas[:, 2].min():.4f}, {all_deltas[:, 2].max():.4f}]")
        LOG.info("=" * 80)
    
    return all_deltas


def diagnose_data_structure(carla_data_dir: Path, num_samples: int = 5) -> None:
    """
    Diagnose the data structure by examining a few sample meta files.
    
    Args:
        carla_data_dir: Root directory containing CARLA expert data
        num_samples: Number of directories to sample
    """
    LOG.info("")
    LOG.info("=" * 80)
    LOG.info("DATA STRUCTURE DIAGNOSIS")
    LOG.info("=" * 80)
    
    metas_dirs = sorted([d for d in carla_data_dir.rglob("metas") if d.is_dir()])
    
    if not metas_dirs:
        LOG.error("No metas directories found!")
        return
    
    # Sample evenly across directories
    sample_indices = np.linspace(0, len(metas_dirs) - 1, min(num_samples, len(metas_dirs)), dtype=int)
    
    for idx in sample_indices:
        metas_dir = metas_dirs[idx]
        meta_files = sorted(metas_dir.glob("*.pkl"))
        
        LOG.info(f"\nDirectory: {metas_dir.relative_to(carla_data_dir)}")
        LOG.info(f"  Total meta files: {len(meta_files)}")
        
        if len(meta_files) > 0:
            # Check first file
            first_file = meta_files[0]
            try:
                meta_dict = load_meta_file(first_file)
                
                LOG.info(f"  First file ({first_file.name}):")
                LOG.info(f"    Keys: {list(meta_dict.keys())}")
                
                if "pos_global" in meta_dict:
                    LOG.info(f"    pos_global: {meta_dict['pos_global']} (type: {type(meta_dict['pos_global'])})")
                else:
                    LOG.warning(f"    'pos_global' NOT FOUND")
                
                if "theta" in meta_dict:
                    LOG.info(f"    theta: {meta_dict['theta']} (type: {type(meta_dict['theta'])})")
                else:
                    LOG.warning(f"    'theta' NOT FOUND")
                
                if "speed" in meta_dict:
                    LOG.info(f"    speed: {meta_dict['speed']}")
                
                # Try to extract pose
                pose = extract_deltas_from_meta(meta_dict)
                if pose is not None:
                    LOG.info(f"    ✓ Successfully extracted pose: {pose}")
                else:
                    LOG.warning(f"    ✗ Failed to extract pose")
                    
            except Exception as e:
                LOG.error(f"    Error reading file: {e}")
        
        # Check if trajectory has enough frames
        if len(meta_files) > 1:
            LOG.info(f"  ✓ Trajectory has {len(meta_files)} frames (sufficient for deltas)")
        elif len(meta_files) == 1:
            LOG.warning(f"  ✗ Trajectory has only 1 frame (NOT sufficient for deltas)")
        else:
            LOG.warning(f"  ✗ Trajectory is empty")
    
    LOG.info("=" * 80)


if __name__ == "__main__":
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print detailed debug information"
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Run diagnostic mode to check data structure (no extraction)"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        LOG.setLevel(logging.DEBUG)
    elif args.verbose:
        LOG.setLevel(logging.INFO)
    
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
                if args.verbose:
                    LOG.info(f"Found CARLA data at: {path}")
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
        raise SystemExit(1)
    
    carla_data_root = Path(carla_data_root)
    
    # Diagnostic mode
    if args.diagnose:
        LOG.info(f"Diagnostic mode: Examining data structure at {carla_data_root}")
        diagnose_data_structure(carla_data_root)
        raise SystemExit(0)
    
    if args.verbose:
        LOG.info("")
        LOG.info("=" * 80)
        LOG.info("LEAD CARLA DELTA EXTRACTION")
        LOG.info("=" * 80)
        LOG.info(f"LEAD root: {lead_root}")
        LOG.info(f"Output directory: {data_dir}")
        LOG.info(f"CARLA data root: {carla_data_root}")
        LOG.info(f"Output file: {output_path}")
        LOG.info("")
    
    # Extract deltas
    LOG.info("Extracting deltas from CARLA expert data...")
    deltas = extract_all_deltas_from_directory(carla_data_root, verbose=args.verbose)
    
    if len(deltas) == 0:
        LOG.error("No deltas extracted! Possible issues:")
        LOG.error("  - All trajectories have only 1 frame (check LEAD_CARLA_DATA_ROOT)")
        LOG.error("  - pose_global or theta fields missing from meta files")
        LOG.error("  - Meta pickle files are corrupted")
        LOG.error("")
        LOG.error("Try running diagnostic mode to check data structure:")
        LOG.error("  python lead/scripts/extract_lead_carla_deltas.py --diagnose")
        raise SystemExit(1)
    
    # Save deltas
    LOG.info("")
    LOG.info(f"Saving {len(deltas)} deltas to {output_path}")
    np.save(str(output_path), deltas)
    LOG.info(f"✓ Successfully saved to {output_path}")
    
    if args.verbose:
        LOG.info("")
        LOG.info("Next step: Build vocabulary with:")
        LOG.info(f"  python lead/scripts/build_kdisks_carla.py --codebook_size 4096 --verbose")
        LOG.info("")
