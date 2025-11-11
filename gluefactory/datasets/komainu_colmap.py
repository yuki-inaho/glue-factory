"""
Komainu COLMAP dataset loader.

This dataset loader reads komainu_colmap data (30 images with COLMAP reconstruction)
and generates image pairs based on covisibility.

Inherits from ColmapImagePairsDataset to reuse COLMAP pair extraction logic.
"""

import logging
from pathlib import Path

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
