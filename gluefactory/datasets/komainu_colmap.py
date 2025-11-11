"""
Komainu COLMAP dataset loader.

This dataset loader reads komainu_colmap data (30 images with COLMAP reconstruction)
and generates image pairs based on covisibility.

Inherits from ColmapImagePairsDataset to reuse COLMAP pair extraction logic.
"""

import logging
from pathlib import Path

import numpy as np
import pycolmap
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
        5. Cache COLMAP reconstructions for efficient access in __getitem__
        """
        # Call parent class _init, which handles all the logic
        super()._init(conf)

        # Cache COLMAP reconstructions for each scene
        # This allows us to access 3D points in __getitem__ for sparse depth generation
        self.reconstructions = {}
        for scene in self.scenes:
            scene_sfm_path = self.root / conf.sfm.format(scene=scene)
            self.reconstructions[scene] = pycolmap.Reconstruction(scene_sfm_path)

        logger.info(
            f"Initialized Komainu COLMAP dataset with {len(self)} pairs "
            f"from {len(self.scenes)} scenes."
        )

    def _generate_sparse_depth(
        self,
        reconstruction: pycolmap.Reconstruction,
        image_name: str,
        keypoints: torch.Tensor,
        K: torch.Tensor,
        T_world2cam,
        threshold_pixels: float = 5.0,
    ):
        """
        Generate sparse depth map at keypoint locations from COLMAP 3D points.

        Args:
            reconstruction: COLMAP Reconstruction object
            image_name: Name of the image (e.g., "0001.jpg")
            keypoints: Keypoint positions [N, 2]
            K: Camera intrinsic matrix [3, 3]
            T_world2cam: Pose object for world-to-camera transformation
            threshold_pixels: Maximum distance (in pixels) to assign depth to keypoint

        Returns:
            depth_keypoints: Depth values at keypoints [N, 1]
            valid_depth_keypoints: Boolean mask indicating valid depths [N, 1]
        """
        # Find image ID from name
        image_id = None
        image_obj = None
        for img_id, img in reconstruction.images.items():
            if img.name == image_name:
                image_id = img_id
                image_obj = img
                break

        if image_id is None:
            logger.warning(f"Image {image_name} not found in reconstruction")
            # Return zero depths
            return (
                torch.zeros(len(keypoints), 1, dtype=torch.float32),
                torch.zeros(len(keypoints), 1, dtype=torch.bool),
            )

        # Collect 3D points observed in this image
        projected_pixels = []
        depths = []

        for p2D in image_obj.points2D:
            if not p2D.has_point3D():
                continue

            point3D_id = p2D.point3D_id
            if point3D_id not in reconstruction.points3D:
                continue

            # Get 3D point in world coordinates
            p3D_world = reconstruction.points3D[point3D_id].xyz  # [3]
            p3D_world_tensor = torch.tensor(p3D_world, dtype=torch.float32)

            # Transform to camera coordinates
            p3D_cam = T_world2cam.transform(p3D_world_tensor)  # [3]
            depth = p3D_cam[2].item()

            # Skip points behind the camera
            if depth <= 0:
                continue

            # Project to pixel coordinates
            p_normalized = p3D_cam[:2] / depth  # [2]
            pixel_h = torch.cat([p_normalized, torch.ones(1)])  # [3]
            pixel = K @ pixel_h  # [3]
            pixel_uv = pixel[:2]  # [2]

            projected_pixels.append(pixel_uv)
            depths.append(depth)

        if len(projected_pixels) == 0:
            logger.warning(
                f"No valid 3D points found for image {image_name}"
            )
            return (
                torch.zeros(len(keypoints), 1, dtype=torch.float32),
                torch.zeros(len(keypoints), 1, dtype=torch.bool),
            )

        # Convert to tensors
        projected_pixels = torch.stack(projected_pixels)  # [M, 2]
        depths_tensor = torch.tensor(depths, dtype=torch.float32)  # [M]

        # For each keypoint, find nearest projected 3D point
        depth_keypoints = torch.zeros(len(keypoints), 1, dtype=torch.float32)
        valid_depth_keypoints = torch.zeros(len(keypoints), 1, dtype=torch.bool)

        for i, kp in enumerate(keypoints):
            # Compute distances to all projected points
            distances = torch.norm(projected_pixels - kp.unsqueeze(0), dim=-1)  # [M]
            min_dist, min_idx = torch.min(distances, dim=0)

            # If within threshold, assign depth
            if min_dist.item() < threshold_pixels:
                depth_keypoints[i, 0] = depths_tensor[min_idx]
                valid_depth_keypoints[i, 0] = True

        return depth_keypoints, valid_depth_keypoints

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

        # Generate sparse depth maps from COLMAP 3D points
        # Extract image names
        image_names = data["name"].split("/")
        name0, name1 = image_names[0], image_names[1]

        # Get reconstruction for this scene
        scene = data["scene"]
        reconstruction = self.reconstructions[scene]

        # Get keypoints (generated by SuperPoint in parent class)
        # Check both possible locations
        kp0 = data.get("keypoints0", data.get("view0", {}).get("keypoints"))
        kp1 = data.get("keypoints1", data.get("view1", {}).get("keypoints"))

        # depth_matcher expects keypoints at top level, so copy them if they exist in views
        if kp0 is not None and "keypoints0" not in data:
            data["keypoints0"] = kp0
        if kp1 is not None and "keypoints1" not in data:
            data["keypoints1"] = kp1

        # Get world-to-camera transforms
        # Pose objects stored in view0/view1
        T_world2cam0 = data["view0"]["T_w2cam"]
        T_world2cam1 = data["view1"]["T_w2cam"]

        # Generate sparse depth for view0
        if kp0 is not None and len(kp0) > 0:
            depth_kp0, valid_kp0 = self._generate_sparse_depth(
                reconstruction=reconstruction,
                image_name=name0,
                keypoints=kp0,
                K=K0,
                T_world2cam=T_world2cam0,
                threshold_pixels=5.0,
            )
            data["depth_keypoints0"] = depth_kp0
            data["valid_depth_keypoints0"] = valid_kp0
        else:
            # No keypoints, return empty tensors
            data["depth_keypoints0"] = torch.zeros(0, 1, dtype=torch.float32)
            data["valid_depth_keypoints0"] = torch.zeros(0, 1, dtype=torch.bool)

        # Generate sparse depth for view1
        if kp1 is not None and len(kp1) > 0:
            depth_kp1, valid_kp1 = self._generate_sparse_depth(
                reconstruction=reconstruction,
                image_name=name1,
                keypoints=kp1,
                K=K1,
                T_world2cam=T_world2cam1,
                threshold_pixels=5.0,
            )
            data["depth_keypoints1"] = depth_kp1
            data["valid_depth_keypoints1"] = valid_kp1
        else:
            # No keypoints, return empty tensors
            data["depth_keypoints1"] = torch.zeros(0, 1, dtype=torch.float32)
            data["valid_depth_keypoints1"] = torch.zeros(0, 1, dtype=torch.bool)

        return data
