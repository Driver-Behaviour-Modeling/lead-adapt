"""
Build K-disks vocabulary from extracted LEAD CARLA deltas.

Runs greedy clustering on motion deltas to create a fixed vocabulary
of motion primitives for the ADAPT model.

Usage:
    python lead/scripts/build_kdisks_carla.py --codebook_size 4096 --verbose
"""

import argparse
import numpy as np
from pathlib import Path
import logging
import sys

# Add lead module to path
lead_root = Path(__file__).parent.parent
sys.path.insert(0, str(lead_root))

from lead.kdisks import kdisks_cluster_deltas, save_kdisks_vocabulary

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
LOG = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Build K-disks vocabulary from CARLA expert motion deltas"
    )
    parser.add_argument(
        "--codebook_size",
        type=int,
        default=4096,
        help="Size of codebook (vocabulary size)"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Distance threshold for cluster membership (meters)"
    )
    parser.add_argument(
        "--heading_weight",
        type=float,
        default=1.0,
        help="Weight for heading in distance metric"
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
    lead_root = Path(__file__).parent.parent
    adapt_dir = lead_root / "lead" / "adapt"
    data_dir = adapt_dir / "data"
    codebook_dir = adapt_dir / "codebooks"
    
    # Create output directory
    codebook_dir.mkdir(parents=True, exist_ok=True)
    
    # Input/output paths
    deltas_path = data_dir / "carla_deltas.npy"
    vocab_path = codebook_dir / "kdisks_carla.pkl"
    
    if args.verbose:
        LOG.info(f"LEAD root: {lead_root}")
        LOG.info(f"Deltas path: {deltas_path}")
        LOG.info(f"Vocabulary path: {vocab_path}")
    
    # Check if deltas exist
    if not deltas_path.exists():
        LOG.error(f"Deltas file not found: {deltas_path}")
        LOG.error("Run extract_lead_carla_deltas.py first")
        return
    
    # Load deltas
    LOG.info(f"Loading deltas from {deltas_path}...")
    deltas = np.load(str(deltas_path))
    LOG.info(f"Loaded {len(deltas)} deltas with shape {deltas.shape}")
    
    if args.verbose:
        LOG.info(f"Delta statistics:")
        LOG.info(f"  Δx: mean={deltas[:, 0].mean():.4f}, std={deltas[:, 0].std():.4f}")
        LOG.info(f"  Δy: mean={deltas[:, 1].mean():.4f}, std={deltas[:, 1].std():.4f}")
        LOG.info(f"  Δh: mean={deltas[:, 2].mean():.4f}, std={deltas[:, 2].std():.4f}")
    
    # Run K-disks clustering
    LOG.info(f"Running K-disks clustering...")
    LOG.info(f"  Codebook size: {args.codebook_size}")
    LOG.info(f"  Tolerance: {args.tolerance}")
    LOG.info(f"  Heading weight: {args.heading_weight}")
    LOG.info(f"  Seed: {args.seed}")
    
    centroids, info = kdisks_cluster_deltas(
        deltas,
        num_clusters=args.codebook_size,
        tolerance=args.tolerance,
        heading_weight=args.heading_weight,
        seed=args.seed
    )
    
    # Prepare config
    config = {
        'codebook_size': args.codebook_size,
        'tolerance': args.tolerance,
        'heading_weight': args.heading_weight,
        'seed': args.seed,
        'source': 'lead_carla_expert',
        'deltas_file': str(deltas_path),
    }
    
    # Save vocabulary
    LOG.info(f"Saving vocabulary...")
    save_kdisks_vocabulary(str(vocab_path), centroids, info, config)
    
    if args.verbose:
        LOG.info(f"Clustering statistics:")
        LOG.info(f"  Centroids created: {len(centroids)}")
        LOG.info(f"  Mean cluster size: {info['cluster_sizes'].mean():.1f}")
        LOG.info(f"  Min cluster size: {info['cluster_sizes'].min()}")
        LOG.info(f"  Max cluster size: {info['cluster_sizes'].max()}")
        LOG.info(f"  Total samples processed: {info['total_samples']}")
        
        LOG.info(f"\nCentroid statistics:")
        LOG.info(f"  Δx: mean={centroids[:, 0].mean():.4f}, std={centroids[:, 0].std():.4f}")
        LOG.info(f"  Δy: mean={centroids[:, 1].mean():.4f}, std={centroids[:, 1].std():.4f}")
        LOG.info(f"  Δh: mean={centroids[:, 2].mean():.4f}, std={centroids[:, 2].std():.4f}")
    
    LOG.info(f"Vocabulary building complete!")
    LOG.info(f"Saved to: {vocab_path}")


if __name__ == "__main__":
    main()
