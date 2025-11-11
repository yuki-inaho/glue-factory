"""
Test suite for komainu_colmap dataset loader.

Following TDD approach:
1. Write tests first (Red Phase)
2. Implement minimum code to pass (Green Phase)
3. Refactor while keeping tests green (Refactor Phase)
"""

import pytest
import torch
from pathlib import Path
from omegaconf import OmegaConf

from gluefactory import settings


class TestKomainuColmapDataset:
    """Test komainu_colmap dataset initialization and data loading."""

    @pytest.fixture
    def dataset_config(self):
        """Provide basic configuration for komainu_colmap dataset."""
        return {
            "name": "komainu_colmap",
            "root": "komainu_colmap",
            "scene_list": ["."],  # Treat root as single scene
            "sfm": "sparse/0",
            "image_dir": "images",
            "views": "views.txt",
            "min_covisible_points": 10,
            "min_overlap": 0.1,
            "max_overlap": 0.9,
            "max_per_scene": 10,  # Small number for testing
            "preprocessing": {
                "resize": 640,
                "side": "long",
                "square_pad": False,
            },
            "num_workers": 0,
            "seed": 42,
            "overwrite": False,
        }

    def test_dataset_init(self, dataset_config):
        """Test 1: Dataset can be instantiated with basic config."""
        from gluefactory.datasets.komainu_colmap import KomainuColmapDataset

        # This should not raise any exceptions
        dataset = KomainuColmapDataset(dataset_config)

        assert dataset is not None
        assert hasattr(dataset, "root")
        assert hasattr(dataset, "views")

    def test_views_txt_generation(self, dataset_config):
        """Test 2: views.txt is generated if it doesn't exist."""
        from gluefactory.datasets.komainu_colmap import KomainuColmapDataset

        # Remove views.txt if it exists to test generation
        views_path = settings.DATA_PATH / dataset_config["root"] / "views.txt"
        if views_path.exists():
            views_path.unlink()

        dataset = KomainuColmapDataset(dataset_config)

        # views.txt should now exist
        assert views_path.exists()

        # Check format: each line should have image_name + pose + camera data
        with open(views_path) as f:
            lines = f.readlines()
            assert len(lines) > 0
            # Format: name R11 R12 ... T1 T2 T3 model width height params...
            first_line = lines[0].strip().split()
            assert len(first_line) >= 15  # name + 9 rotation + 3 translation + model + w + h + params

    def test_covisible_pairs_extraction(self, dataset_config):
        """Test 3: Covisible pairs are extracted correctly."""
        from gluefactory.datasets.komainu_colmap import KomainuColmapDataset

        dataset = KomainuColmapDataset(dataset_config)

        # Pairs file should be generated
        pairs_pattern = f"*pairs_{dataset_config['min_covisible_points']}-*.txt"
        pairs_files = list((settings.DATA_PATH / dataset_config["root"] / "covisibility").glob(pairs_pattern))

        assert len(pairs_files) > 0, "No covisibility pairs file found"

        # Check pairs format
        with open(pairs_files[0]) as f:
            lines = f.readlines()
            assert len(lines) > 0
            # Format: image0.jpg image1.jpg
            first_line = lines[0].strip().split()
            assert len(first_line) == 2

    def test_dataset_len(self, dataset_config):
        """Test 4: Dataset length is correct."""
        from gluefactory.datasets.komainu_colmap import KomainuColmapDataset

        dataset = KomainuColmapDataset(dataset_config)

        # Should have at least 1 pair, but limited by max_per_scene
        assert len(dataset) > 0
        assert len(dataset) <= dataset_config["max_per_scene"]

    def test_dataset_getitem(self, dataset_config):
        """Test 5: __getitem__ returns correct data structure."""
        from gluefactory.datasets.komainu_colmap import KomainuColmapDataset

        dataset = KomainuColmapDataset(dataset_config)

        # Get first item
        data = dataset[0]

        # Check required keys
        assert "view0" in data
        assert "view1" in data
        assert "T_0to1" in data
        assert "name" in data

        # Check view0 structure
        assert "image" in data["view0"]
        assert "camera" in data["view0"]
        assert "T_w2cam" in data["view0"]

        # Check view1 structure
        assert "image" in data["view1"]
        assert "camera" in data["view1"]
        assert "T_w2cam" in data["view1"]

        # Check data types
        assert isinstance(data["view0"]["image"], torch.Tensor)  # Tensor
        assert isinstance(data["T_0to1"], object)  # Pose object

    def test_image_preprocessing(self, dataset_config):
        """Test 6: Images are preprocessed correctly."""
        from gluefactory.datasets.komainu_colmap import KomainuColmapDataset

        dataset = KomainuColmapDataset(dataset_config)
        data = dataset[0]

        # Images should be resized according to config
        # Image shape: [C, H, W]
        img0 = data["view0"]["image"]
        assert img0.ndim == 3, f"Expected 3D tensor, got shape {img0.shape}"
        assert img0.shape[0] == 3 or img0.shape[0] == 1, f"Expected RGB or grayscale, got {img0.shape[0]} channels"  # RGB or grayscale

        # Check that longer side is resized to 640
        max_side = max(img0.shape[1], img0.shape[2])
        # Allow some tolerance due to different resize strategies
        assert abs(max_side - dataset_config["preprocessing"]["resize"]) < 50

    def test_relative_pose_computation(self, dataset_config):
        """Test 7: Relative pose T_0to1 is computed correctly."""
        from gluefactory.datasets.komainu_colmap import KomainuColmapDataset

        dataset = KomainuColmapDataset(dataset_config)
        data = dataset[0]

        # T_0to1 should transform points from cam0 to cam1
        # T_0to1 = T_w2cam1 @ T_cam02w = T_w2cam1 @ inv(T_w2cam0)
        T_0to1 = data["T_0to1"]
        T_w2cam0 = data["view0"]["T_w2cam"]  # Already batched
        T_w2cam1 = data["view1"]["T_w2cam"]  # Already batched

        # Verify the relationship (approximately, due to numerical precision)
        expected_T_0to1 = T_w2cam1 @ T_w2cam0.inv()

        # Check that rotation and translation are close
        assert torch.allclose(T_0to1.data_, expected_T_0to1.data_, atol=1e-4)

    def test_homography_computation(self, dataset_config):
        """Test 8: Homography matrix H_0to1 is computed correctly."""
        from gluefactory.datasets.komainu_colmap import KomainuColmapDataset

        dataset = KomainuColmapDataset(dataset_config)
        data = dataset[0]

        # H_0to1 should be present
        assert "H_0to1" in data, "H_0to1 key should be present in data"

        # H_0to1 should be a 3x3 matrix
        H = data["H_0to1"]
        assert isinstance(H, torch.Tensor), f"H_0to1 should be a tensor, got {type(H)}"
        assert H.shape == (3, 3), f"H_0to1 should be 3x3, got shape {H.shape}"

        # H should have reasonable values (not all zeros, not NaN/Inf)
        assert not torch.isnan(H).any(), "H_0to1 contains NaN values"
        assert not torch.isinf(H).any(), "H_0to1 contains Inf values"
        assert torch.abs(H).sum() > 0, "H_0to1 is all zeros"

        # H should be approximately invertible (det != 0)
        det = torch.det(H)
        assert torch.abs(det) > 1e-6, f"H_0to1 determinant too close to zero: {det}"

        # Test homography consistency: points from view0 should map to view1
        # Get some keypoints
        kp0 = data.get("keypoints0")
        kp1 = data.get("keypoints1")

        if kp0 is not None and kp1 is not None and len(kp0) > 0:
            # Convert keypoints to homogeneous coordinates
            kp0_h = torch.cat([kp0[:10], torch.ones(min(10, len(kp0)), 1)], dim=-1)  # [N, 3]

            # Apply homography
            kp0_warped = (H @ kp0_h.T).T  # [N, 3]
            kp0_warped = kp0_warped[:, :2] / kp0_warped[:, 2:3]  # Normalize to [N, 2]

            # The warped points should be within image bounds
            # (not a strong test, but checks basic sanity)
            img_shape = data["view1"]["image"].shape
            assert kp0_warped[:, 0].min() > -img_shape[2], "Warped x-coords too negative"
            assert kp0_warped[:, 1].min() > -img_shape[1], "Warped y-coords too negative"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
