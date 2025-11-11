# COLMAP Reconstruction Data

This directory contains COLMAP 3D reconstruction output data.

## Directory Structure

```
komainu_colmap/
├── cameras.txt       # Camera calibration parameters
├── images.txt        # Image poses and feature observations
├── points3D.txt      # 3D point cloud data
├── frames.txt        # Frame poses with rig transformations
├── rigs.txt          # Rig configuration
├── database.db       # COLMAP SQLite database
└── project.ini       # COLMAP project configuration
```

## Data Details

### cameras.txt
Camera calibration data containing intrinsic parameters.

- **Number of cameras**: 1
- **Camera model**: SIMPLE_PINHOLE
- **Image dimensions**: 720 x 1280 pixels
- **Focal length**: 1167.50 pixels
- **Principal point**: (360, 640)

Format:
```
CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
1 SIMPLE_PINHOLE 720 1280 1167.5026006651085 360 640
```

### images.txt
Image registration data with camera poses and feature point observations.

- **Number of images**: 30
- **Mean observations per image**: 1806.23 feature points
- **File size**: Approximately 9.9 MB

Format:
```
IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
POINTS2D[] as (X, Y, POINT3D_ID)
```

Each image entry contains:
- Rotation quaternion (QW, QX, QY, QZ)
- Translation vector (TX, TY, TZ)
- Camera ID reference
- List of 2D feature points and their corresponding 3D point IDs

### points3D.txt
3D point cloud reconstruction data with color and track information.

- **Number of 3D points**: 10,771
- **Mean track length**: 5.03 observations per point
- **File size**: Approximately 1.4 MB

Format:
```
POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
```

Each point contains:
- 3D position coordinates (X, Y, Z)
- RGB color values (R, G, B)
- Reprojection error
- Track information showing which images observe this point

### frames.txt
Frame pose data using rig-based representation.

- **Number of frames**: 30
- **Rig configuration**: Single camera rig

Format:
```
FRAME_ID, RIG_ID, RIG_FROM_WORLD[QW, QX, QY, QZ, TX, TY, TZ], NUM_DATA_IDS, DATA_IDS[]
```

Each frame represents a timestamp with:
- Rig pose transformation (quaternion + translation)
- Associated sensor data (CAMERA 1 for all frames)

### rigs.txt
Rig calibration configuration.

- **Number of rigs**: 1
- **Reference sensor**: CAMERA 1
- **Configuration**: Single camera setup with no additional sensors

### database.db
SQLite database containing:
- Raw feature extraction data
- Feature matching results
- Geometric verification data
- Camera and image metadata

This database is used internally by COLMAP during the reconstruction pipeline.

### project.ini
COLMAP project configuration file containing all processing parameters:

- **Source data**:
  - Database: `/home/inaho-omen/data/colmap/database.db`
  - Images: `/home/inaho-omen/data/colmap/images`

- **Feature extraction settings**:
  - SIFT features with GPU acceleration
  - Maximum 8,192 features per image
  - Maximum image size: 3,200 pixels

- **Matching settings**:
  - Sequential matching with overlap of 10 images
  - GPU-accelerated matching
  - Cross-check enabled

- **Reconstruction settings**:
  - Bundle adjustment with focal length refinement
  - Multiple models enabled
  - Minimum model size: 10 images

## Reconstruction Summary

This dataset represents a sequential video capture processed through COLMAP's Structure-from-Motion pipeline:

- 30 input frames from video sequence
- Single camera with SIMPLE_PINHOLE model
- Successfully reconstructed 10,771 3D points
- Average of 1,806 feature observations per image
- Mean track length of 5.03 images per 3D point

The reconstruction appears to be a continuous camera motion path, likely from a handheld or moving video capture of a scene.

## Usage

This data can be used with:
- COLMAP for further processing or refinement
- Visualization tools that support COLMAP format
- Dense reconstruction pipelines (MVS)
- Neural rendering methods (NeRF, Gaussian Splatting)
