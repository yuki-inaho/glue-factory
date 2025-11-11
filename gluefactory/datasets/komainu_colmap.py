"""
Komainu COLMAP dataset loader.

This dataset loader reads komainu_colmap data (30 images with COLMAP reconstruction)
and generates image pairs based on covisibility.

Inherits from ColmapImagePairsDataset to reuse COLMAP pair extraction logic.
"""

import logging
from pathlib import Path

import numpy as np
import torch

from .pairs_from_colmap import ColmapImagePairsDataset

logger = logging.getLogger(__name__)


class KomainuColmapDataset(ColmapImagePairsDataset):
    """
    Dataset loader for komainu_colmap.

    This dataset contains 30 images captured with a SIMPLE_PINHOLE camera,
    with full COLMAP reconstruction (poses, 3D points, covisibility graph).

    Attributes:
        root: Path to komainu_colmap directory
        sfm: Path to COLMAP sparse reconstruction (relative to root)
        views: Path to views.txt (will be auto-generated if not exists)
        min_covisible_points: Minimum number of shared 3D points for a valid pair
        min_overlap: Minimum overlap ratio (0.0-1.0)
        max_overlap: Maximum overlap ratio (0.0-1.0)
        max_per_scene: Maximum number of pairs to sample per scene
    """

    default_conf = {
        **ColmapImagePairsDataset.default_conf,
        "root": "komainu_colmap",
        "sfm": "sparse/0",
        "image_dir": "images",
        "views": "views.txt",
        "min_covisible_points": 10,
        "min_overlap": 0.1,
        "max_overlap": 0.9,
        "max_per_scene": None,  # None = use all pairs
        "scene_list": None,  # None = use all scenes (only 1 scene for komainu)
        "preprocessing": {
            "resize": 640,
            "side": "long",
            "square_pad": False,
            "antialias": False,
        },
        "num_workers": 2,
        "seed": 42,
        "overwrite": False,  # If True, regenerate views.txt and pairs
    }

    def _init(self, conf):
        """
        Initialize the dataset.

        This will:
        1. Load COLMAP reconstruction from sfm path
        2. Generate views.txt if it doesn't exist
        3. Extract covisible pairs
        4. Generate pair lists based on overlap criteria
        """
        # Call parent class _init, which handles all the logic
        super()._init(conf)

        logger.info(
            f"Initialized Komainu COLMAP dataset with {len(self)} pairs "
            f"from {len(self.scenes)} scenes."
        )

    def __getitem__(self, idx):
        """
        Get a data sample including homography matrix H_0to1.

        Computes homography from COLMAP poses using a planar scene assumption.
        H = K1 @ (R - t @ n^T / d) @ K0^{-1}
        where n is the plane normal and d is the distance from origin.
        """
        # Get base data from parent class
        data = super().__getitem__(idx)

        # Extract camera matrices and poses
        K0 = data["view0"]["camera"].K  # [3, 3]
        K1 = data["view1"]["camera"].K  # [3, 3]
        T_0to1 = data["T_0to1"]  # Pose object

        # Extract rotation and translation from Pose
        # Pose.data_ contains [qw, qx, qy, qz, tx, ty, tz, ...]
        R = T_0to1.R  # Rotation matrix [3, 3]
        t = T_0to1.t  # Translation vector [3]

        # Planar scene assumption: use z=0 plane (ground plane)
        # Plane normal pointing up (camera looks down at ground)
        n = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)  # [3]

        # Estimate plane distance d using median depth of 3D points if available
        # For simplicity, use a fixed distance assumption
        # TODO: Could improve by computing median depth from covisible 3D points
        d = 2.0  # Assume scene is ~2 meters away (reasonable for indoor/close-range)

        # Compute homography: H = K1 @ (R - t @ n^T / d) @ K0^{-1}
        # n^T: [1, 3], t: [3, 1] -> t @ n^T: [3, 3]
        t_n = torch.outer(t, n)  # [3, 3]
        H_normalized = R - t_n / d  # [3, 3]

        # Apply camera matrices
        K0_inv = torch.linalg.inv(K0)
        H = K1 @ H_normalized @ K0_inv  # [3, 3]

        # Normalize homography (optional, but good practice)
        H = H / H[2, 2]

        data["H_0to1"] = H

        return data
