# brain_utility

A Python toolkit for preprocessing, coregistering, and evaluating brain MRI volumes (NIfTI format).

## Modules

| Module | Purpose |
|--------|---------|
| `rescaling` | Normalize voxel intensities to a target range |
| `preprocessing` | Pair longitudinal scans, match histograms, extract bounding boxes, convert formats |
| `dm_preprocessing` | Build difference-map pairs from longitudinal sequences |
| `coregister` | Rigid-body coregistration of T1/T2-FLAIR volumes using SimpleITK |
| `mask_coregistering` | Coregister image + segmentation mask pairs with a shared transform |
| `upsampling` | Resample volumes to a target voxel grid |
| `evaluation` | Compute MAE, MSE, PSNR, SSIM, reconstruction and evolution scores |

## Install

```bash
pip install -e ".[dev]"
```

## Test

```bash
pytest
```
