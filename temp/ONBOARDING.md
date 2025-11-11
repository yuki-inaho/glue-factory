# ONBOARDING - glue-factory komainu_colmap Training Implementation

## Project Information

- **Repository**: yuki-inaho/glue-factory
- **Branch**: `claude/update-readme-docs-011CV1sZs4vLP9sAsAvVsCmw`
- **Date Started**: 2025-11-11
- **Primary Goal**: Implement training pipeline for komainu_colmap dataset using TDD methodology

## Current Status

### Completed Work

1. **Environment Setup** ✅
   - UV package manager configured
   - 119 packages installed (108 base + 5 extra + dev dependencies)
   - Fixed missing dependencies: tensordict, plotly
   - All imports verified working

2. **Dataset Implementation** ✅
   - Created `gluefactory/datasets/komainu_colmap.py`
   - Inherits from ColmapImagePairsDataset
   - Loads 30 images from komainu_colmap COLMAP reconstruction
   - Generates covisibility pairs with 10,771 3D points

3. **Test Suite** ✅
   - Created `tests/test_komainu_colmap.py`
   - 7 comprehensive tests following TDD approach
   - All tests passing (7/7)
   - Tests cover: initialization, views.txt generation, pairs extraction, data loading, preprocessing, pose computation

4. **Configuration Files** ✅
   - Created `gluefactory/configs/data/komainu_colmap.yaml`
   - Created `gluefactory/configs/komainu_train_homography.yaml`
   - Uses homography-based ground truth (depth maps not yet available)

5. **Bug Fixes** ✅
   - Fixed PyTorch 2.1.2 compatibility issue with torch.compiler.set_stance
   - Fixed scene_list configuration to treat root as single scene
   - Fixed test assertions for correct tensor structure

### In Progress

- **Training Configuration**: Missing complete train section (seed, lr_schedule, etc.)
- **Training Execution**: Not yet tested due to config issues

### Blocked/Pending

- Run dry run training (0 epochs) to verify pipeline
- Run 1 epoch training to confirm full pipeline works
- Add code formatting and linting
- Update README.md with komainu_colmap usage

## Environment Setup

### Prerequisites

- Python 3.10+
- UV package manager
- CUDA-capable GPU (recommended)

### Setup Steps

```bash
# Install UV if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync --extra extra --dev

# Verify installation
uv run python -c "import gluefactory; print('OK')"
```

### Key Dependencies

- PyTorch 2.1.2 with CUDA
- pycolmap 0.6.1
- tensordict (custom batching)
- plotly 6.4.0 (visualization)
- pytest (testing)
- hydra-core (configuration)

## Project Structure

### Key Files

```
gluefactory/
├── datasets/
│   └── komainu_colmap.py          # Dataset loader (NEW)
├── configs/
│   ├── data/
│   │   └── komainu_colmap.yaml    # Dataset config (NEW)
│   └── komainu_train_homography.yaml  # Training config (NEW)
├── trainer.py                      # Modified for PyTorch 2.1 compat
tests/
└── test_komainu_colmap.py         # Test suite (NEW)
temp/
├── workdoc_komainu_training.md    # TDD work plan
├── workdoc_nov11.md               # Environment setup log
└── ONBOARDING.md                   # This file
data/
└── komainu_colmap/                # Dataset (30 images)
    ├── images/
    ├── sparse/0/
    ├── views.txt                   # Auto-generated
    └── covisibility/               # Auto-generated pairs
```

### Dataset Structure

The komainu_colmap dataset contains:
- 30 RGB images in `data/komainu_colmap/images/`
- COLMAP sparse reconstruction in `data/komainu_colmap/sparse/0/`
- 10,771 3D points
- Automatically generated view poses and covisibility pairs

## Running Tests

```bash
# Run all komainu_colmap tests
uv run pytest tests/test_komainu_colmap.py -v

# Run specific test
uv run pytest tests/test_komainu_colmap.py::TestKomainuColmapDataset::test_dataset_init -v

# Run with output
uv run pytest tests/test_komainu_colmap.py -v -s
```

Expected output: 7 passed tests

## Training

### Current Configuration

The training uses:
- **Extractor**: SuperPoint (frozen, pretrained)
- **Matcher**: LightGlue (trainable)
- **Ground Truth**: Homography matcher
- **Dataset**: komainu_colmap (237 pairs, limited to config max)

### Training Command (Not Yet Working)

```bash
# Dry run (0 epochs) - BLOCKED: Missing train.seed config
uv run python -m gluefactory.train komainu_train_homography train.epochs=0

# Full training (once config fixed)
uv run python -m gluefactory.train komainu_train_homography
```

### Known Issues with Training

1. **Missing train.seed**: Configuration needs complete train section
   - Need to add: seed, lr, lr_schedule, log_every_iter, eval_every_iter
   - Reference: `gluefactory/configs/lightglue_megadepth.yaml`

## Common Issues and Fixes

### Issue: ModuleNotFoundError: tensordict
**Fix**: `uv add tensordict` or ensure it's in pyproject.toml

### Issue: ModuleNotFoundError: plotly
**Fix**: `uv add plotly`

### Issue: torch.compiler.set_stance AttributeError
**Fix**: Already patched in trainer.py with conditional decorator

### Issue: NotADirectoryError with README.md/covisibility
**Fix**: Use `scene_list: ["."]` in dataset config to treat root as single scene

### Issue: Test failures with tensor indexing
**Fix**: Data is TensorDict, not batched in lists - access directly without [0]

## Development Methodology

This project follows:
- **TDD (Test-Driven Development)**: Write tests first, then implement
- **t-wada style**: Red → Green → Refactor cycle
- **DRY/KISS/SOLID principles**
- **Action counting**: Track every action, reminder every 20 actions
- **Work documentation**: Maintain detailed logs in temp/workdoc_*.md

## Next Steps

1. **Fix Training Configuration**
   - Add complete train section to komainu_train_homography.yaml
   - Include: seed, epochs, lr, lr_schedule, log_every_iter, eval_every_iter

2. **Verify Training Pipeline**
   - Run dry run (0 epochs)
   - Run 1 epoch training
   - Verify outputs and checkpoints

3. **Code Quality**
   - Add black formatting
   - Add flake8 linting
   - Ensure all code follows style guide

4. **Documentation**
   - Update README.md with komainu_colmap usage
   - Add training examples
   - Document configuration options

## References

- Main work plan: `temp/workdoc_komainu_training.md`
- Environment log: `temp/workdoc_nov11.md`
- Example config: `gluefactory/configs/lightglue_megadepth.yaml`
- Parent dataset: `gluefactory/datasets/colmap_image_pairs.py`

## Contact & Questions

For issues or questions:
- Check work documentation in temp/
- Review test suite for usage examples
- Reference existing configs for pattern matching

---

**Last Updated**: 2025-11-11
**Status**: Dataset and tests complete, training config in progress
