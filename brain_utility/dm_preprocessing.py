import argparse
import itertools
import logging
import os
import shutil
from datetime import datetime
from typing import List, Tuple

import numpy as np
import SimpleITK as sitk

from brain_utility.utils.parser import parse, get_file_paths
from brain_utility.preprocessing import align_images, create_train_val_test_split

logger = logging.getLogger(__name__)


# --- Pure functions ---


def compute_difference_map(
    arr1: np.ndarray, arr2: np.ndarray
) -> np.ndarray:
    return arr2 - arr1


# --- I/O wrappers ---


def create_paired_dm_images(
    data_path: str, mri_type: str, split_perc: float = 0.7
) -> None:
    if mri_type not in ("T1", "T2F"):
        raise ValueError(
            f"Invalid MRI type '{mri_type}': must be 'T1' or 'T2F'"
        )

    counter = 0
    filetree = parse(data_path, MRI_type=mri_type)

    new_dir = os.path.join(data_path, "pairing")
    a_dir = os.path.join(new_dir, "A")
    b_dir = os.path.join(new_dir, "B")
    ab_dir = os.path.join(new_dir, "AB")

    os.makedirs(a_dir, exist_ok=True)
    os.makedirs(b_dir, exist_ok=True)
    os.makedirs(ab_dir, exist_ok=True)

    date_format = "%d.%m.%y"
    for patient, dates in filetree.items():
        logger.info("%s", patient)

        dates_list = sorted(
            dates.keys(),
            key=lambda x: datetime.strptime(x, date_format),
        )
        combs = list(itertools.combinations(dates_list, 3))

        for date1, date2, date3 in combs:
            date1_parsed = datetime.strptime(date1, date_format)
            date2_parsed = datetime.strptime(date2, date_format)
            date3_parsed = datetime.strptime(date3, date_format)

            diff1 = float((date2_parsed - date1_parsed).days)
            diff2 = float((date3_parsed - date2_parsed).days)
            ratio = round(diff2 / diff1, 3)

            img1_path = dates[date1][0]
            img2_path = dates[date2][0]
            img3_path = dates[date3][0]
            logger.info("Date1: %s, Date2: %s, Date3: %s", date1, date2, date3)
            logger.info("img1: %s, img2: %s, img3: %s", img1_path, img2_path, img3_path)

            img1 = sitk.ReadImage(img1_path)
            img2 = sitk.ReadImage(img2_path)
            img3 = sitk.ReadImage(img3_path)

            arr1 = sitk.GetArrayFromImage(img1)
            arr2 = sitk.GetArrayFromImage(img2)

            diff_arr = compute_difference_map(arr1, arr2)
            diff_img = sitk.GetImageFromArray(diff_arr)

            path_a = os.path.join(a_dir, f"{counter}_{ratio}r.nii")
            path_b = os.path.join(b_dir, f"{counter}_{ratio}r.nii")
            try:
                sitk.WriteImage(diff_img, path_a)
                sitk.WriteImage(img3, path_b)
            except IndexError:
                continue

            counter += 1

    a_paths = get_file_paths(a_dir)
    b_paths = get_file_paths(b_dir)
    align_images(a_paths, b_paths, ab_dir)

    try:
        ab_paths = get_file_paths(ab_dir)
        train, val, test = create_train_val_test_split(ab_paths, split_perc)

        train_dir = os.path.join(new_dir, "train")
        val_dir = os.path.join(new_dir, "val")
        test_dir = os.path.join(new_dir, "test")

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        for file in train:
            shutil.move(file, train_dir)
        for file in val:
            shutil.move(file, val_dir)
        for file in test:
            shutil.move(file, test_dir)

        shutil.rmtree(a_dir)
        shutil.rmtree(b_dir)
        shutil.rmtree(ab_dir)
    except ValueError as e:
        logger.error("Could not create train-val-test split: %s", e)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("rootdir", help="Directory of Data")
    args = parser.parse_args()
    create_paired_dm_images(args.rootdir, "T2F")
