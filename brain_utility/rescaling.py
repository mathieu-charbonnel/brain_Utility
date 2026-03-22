import argparse
import logging
import os

import numpy as np
import SimpleITK as sitk

logger = logging.getLogger(__name__)


# --- Pure functions ---


def normalize_array(
    arr: np.ndarray, pixel_min: float, pixel_max: float
) -> np.ndarray:
    half_range = (pixel_max - pixel_min) / 2.0
    midpoint = (pixel_max + pixel_min) / 2.0
    return (arr - midpoint) / half_range


# --- I/O wrappers ---


def rescale(
    data_path: str, pixel_max: float = 1989.0, pixel_min: float = -1000.0
) -> None:
    for root, dirs, files in os.walk(data_path):
        for name in files:
            file_path = os.path.join(root, name)
            logger.info("file: %s", file_path)
            img = sitk.ReadImage(file_path)
            img_arr = sitk.GetArrayFromImage(img)

            img_arr = normalize_array(img_arr, pixel_min, pixel_max)

            logger.info("rescaled")
            new_nii = sitk.GetImageFromArray(img_arr)
            sitk.WriteImage(new_nii, file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("rootdir", help="Directory of Data")
    args = parser.parse_args()
    rescale(args.rootdir)
