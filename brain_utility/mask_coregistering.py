import argparse
import logging
import shutil
from datetime import datetime
from typing import List, Optional, Tuple

import SimpleITK as sitk

from brain_utility.coregister import build_registration_method
from brain_utility.utils.parser import (
    T1_exists,
    parse_tumor_masking,
    segmentation_exists,
)

logger = logging.getLogger(__name__)


# --- Pure functions ---


def register_image_pair(
    fixed: sitk.Image,
    moving: sitk.Image,
    fixed_seg: sitk.Image,
    moving_seg: sitk.Image,
    rmethod: sitk.ImageRegistrationMethod,
) -> Tuple[sitk.Image, sitk.Image]:
    initial_transform = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    rmethod.SetInitialTransform(initial_transform, inPlace=False)
    final_transform = rmethod.Execute(fixed, moving)

    resampled_image = sitk.Resample(
        moving,
        fixed,
        final_transform,
        sitk.sitkLinear,
        0.0,
        moving.GetPixelID(),
    )
    resampled_image = sitk.Cast(resampled_image, sitk.sitkInt16)

    resampled_seg = sitk.Resample(
        moving_seg,
        fixed_seg,
        final_transform,
        sitk.sitkLinear,
        0.0,
        moving_seg.GetPixelID(),
    )
    resampled_seg = sitk.Cast(resampled_seg, sitk.sitkInt16)

    return resampled_image, resampled_seg


# --- I/O wrappers ---


def coregister_masks(data_path: str) -> None:
    rmethod = build_registration_method()

    filetree = parse_tumor_masking(data_path)
    logger.debug("Filetree: %s", filetree)

    date_format = "%d.%m.%y"
    for patient, dates in filetree.items():
        logger.info("Patient: %s", patient)

        dates_list = sorted(
            dates.keys(), key=lambda x: datetime.strptime(x, date_format)
        )
        logger.info("Dates List: %s", dates_list)

        if not dates_list:
            logger.warning("No dates found for patient")
            continue

        first_date = dates_list[0]
        image_dirs = dates[first_date]
        del dates[first_date]
        logger.info("Using as First Date: %s", first_date)

        fixed_path = [x for x in image_dirs if T1_exists(x)][0]
        fixed_seg_path = [x for x in image_dirs if segmentation_exists(x)][0]
        fixed_image = sitk.ReadImage(fixed_path, sitk.sitkFloat32)
        fixed_seg = sitk.ReadImage(fixed_seg_path, sitk.sitkFloat32)
        logger.info("Fixed Image T1: %s", fixed_path)
        logger.info("Fixed segmentation: %s", fixed_seg_path)

        for date, files in dates.items():
            moving_path: Optional[str] = None
            moving_image = None
            moving_segmentation_path: Optional[str] = None
            moving_segmentation = None

            for file in files:
                if T1_exists(file):
                    moving_path = file
                    moving_image = sitk.ReadImage(moving_path, sitk.sitkFloat32)
                if segmentation_exists(file):
                    moving_segmentation_path = file
                    moving_segmentation = sitk.ReadImage(
                        moving_segmentation_path, sitk.sitkFloat32
                    )

            if moving_path is None or moving_segmentation_path is None:
                logger.warning("Missing T1 or segmentation for date: %s", date)
                continue

            logger.info("Moving Image: %s", moving_path)
            logger.info("Moving Segmentation: %s", moving_segmentation_path)

            try:
                resampled_img, resampled_seg = register_image_pair(
                    fixed_image, moving_image,
                    fixed_seg, moving_segmentation,
                    rmethod,
                )
                logger.info("-> coregistered")

                sitk.WriteImage(resampled_img, moving_path)
                logger.info("-> t1 resampled and exported")

                sitk.WriteImage(resampled_seg, moving_segmentation_path)
                logger.info("-> segmentation resampled and exported")

            except RuntimeError as e:
                logger.error(
                    "Could not register image %s: %s", moving_path, e
                )


def main(args: argparse.Namespace) -> None:
    shutil.copytree(args.rootdir, args.destinationdir, dirs_exist_ok=True)
    logger.info("Co-registering images for each patient")
    coregister_masks(args.destinationdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("rootdir", help="Directory of Data")
    parser.add_argument(
        "destinationdir",
        help="Directory where we intend to store the registered data",
    )
    main(parser.parse_args())
