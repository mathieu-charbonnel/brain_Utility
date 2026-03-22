import argparse
import logging
import os
import shutil
import subprocess
from datetime import datetime
from typing import List

import SimpleITK as sitk

from brain_utility.utils.parser import T1_exists, T2F_exists, parse

logger = logging.getLogger(__name__)


def all_sequences_exist(image_dirs: List[str]) -> bool:
    t1_found = False
    t2_found = False
    for image_dir in image_dirs:
        t1_found = t1_found or T1_exists(image_dir)
        t2_found = t2_found or T2F_exists(image_dir)
    return t1_found and t2_found


# --- Pure functions ---


def build_registration_method() -> sitk.ImageRegistrationMethod:
    rmethod = sitk.ImageRegistrationMethod()
    rmethod.SetMetricAsMattesMutualInformation(numberOfHistogramBins=100)
    rmethod.SetMetricSamplingStrategy(rmethod.RANDOM)
    rmethod.SetMetricSamplingPercentage(0.01)
    rmethod.SetInterpolator(sitk.sitkLinear)
    rmethod.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=200,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    rmethod.SetOptimizerScalesFromPhysicalShift()
    rmethod.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    rmethod.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    rmethod.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    return rmethod


def register_image(
    fixed: sitk.Image,
    moving: sitk.Image,
    rmethod: sitk.ImageRegistrationMethod,
) -> sitk.Image:
    initial_transform = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    rmethod.SetInitialTransform(initial_transform, inPlace=False)
    final_transform = rmethod.Execute(fixed, moving)
    resampled = sitk.Resample(
        moving,
        fixed,
        final_transform,
        sitk.sitkLinear,
        0.0,
        moving.GetPixelID(),
    )
    return sitk.Cast(resampled, sitk.sitkInt16)


# --- I/O wrappers ---


def register_to_atlas_flair(data_path: str) -> None:
    script = os.getcwd() + "/utils/pipeline_mask_FLAIR_patients_v2.sh"
    filetree = parse(data_path, MRI_type="T2F")

    for patient, dates in filetree.items():
        logger.info("Patient: %s", patient)

        for date, files in dates.items():
            logger.info("%s", date)
            for file_dir in files:
                if file_dir.endswith(".nii.gz"):
                    file_dir = os.path.splitext(
                        os.path.splitext(file_dir)[0]
                    )[0]
                    ext = "nii.gz"
                elif file_dir.endswith(".nii"):
                    file_dir = os.path.splitext(file_dir)[0]
                    ext = "nii"
                else:
                    continue

                filename = os.path.basename(file_dir)
                directory = os.path.dirname(file_dir)
                logger.info("%s/%s", directory, filename)
                subprocess.check_call(
                    [script, data_path, directory, filename, ext],
                )


def coregister(data_path: str) -> None:
    rmethod = build_registration_method()

    filetree = parse(data_path)
    logger.debug("Filetree: %s", filetree)

    date_format = "%d.%m.%y"
    for patient, dates in filetree.items():
        logger.info("Patient: %s", patient)

        dates_list = sorted(
            dates.keys(), key=lambda x: datetime.strptime(x, date_format)
        )
        logger.info("Dates List: %s", dates_list)

        first_date = None
        image_dirs = None
        for date in dates_list:
            files = dates[date]
            if all_sequences_exist(files):
                first_date, image_dirs = date, files
                break

        if first_date is None:
            logger.warning("No date found with both T1 and T2 sequences")
            continue

        del dates[first_date]
        logger.info("Using as First Date: %s", first_date)

        fixed_path_t1 = [x for x in image_dirs if T1_exists(x)][0]
        fixed_path_t2 = [x for x in image_dirs if T2F_exists(x)][0]
        fixed_image_t1 = sitk.ReadImage(fixed_path_t1, sitk.sitkFloat32)
        fixed_image_t2 = sitk.ReadImage(fixed_path_t2, sitk.sitkFloat32)
        logger.info("Fixed Image T1: %s", fixed_path_t1)
        logger.info("Fixed Image T2: %s", fixed_path_t2)

        for date, files in dates.items():
            for moving_path in files:
                moving_image = sitk.ReadImage(moving_path, sitk.sitkFloat32)
                logger.info("Moving Image: %s", moving_path)

                if T1_exists(moving_path):
                    fixed_image = fixed_image_t1
                else:
                    fixed_image = fixed_image_t2

                try:
                    resampled = register_image(fixed_image, moving_image, rmethod)
                    logger.info("-> coregistered and resampled")

                    sitk.WriteImage(resampled, moving_path)
                    logger.info("-> exported")

                except RuntimeError as e:
                    logger.error(
                        "Could not register image %s: %s", moving_path, e
                    )


def main(args: argparse.Namespace) -> None:
    shutil.copytree(args.rootdir, args.destinationdir, dirs_exist_ok=True)

    if not args.f:
        logger.info("Co-registering images for each patient")
        coregister(args.destinationdir)
    else:
        logger.info("Registering with Flair Atlas")
        register_to_atlas_flair(args.destinationdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("rootdir", help="Directory of Data")
    parser.add_argument(
        "-f",
        action="store_true",
        help="Register to a Flair Atlas and Mask (Flair images only)",
    )
    parser.add_argument(
        "destinationdir",
        help="Directory where we intend to store the registered data",
    )
    main(parser.parse_args())
