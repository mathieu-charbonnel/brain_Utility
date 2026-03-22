import argparse
import itertools
import logging
import os
import shutil
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import SimpleITK as sitk
from skimage.exposure import match_histograms
from sklearn.model_selection import train_test_split

from brain_utility.utils.parser import parse, get_file_paths

logger = logging.getLogger(__name__)


# --- Pure functions ---


def concatenate_images(
    img_a: sitk.Image,
    img_b: sitk.Image,
    size: Tuple[int, int, int] = (128, 128, 128),
) -> sitk.Image:
    img_a = sitk.Cast(img_a, sitk.sitkInt32)
    img_b = sitk.Cast(img_b, sitk.sitkInt32)
    sx, sy, sz = size
    aligned_image = sitk.Image(sx, sy, sz * 2, sitk.sitkInt32)
    aligned_image = sitk.Paste(
        aligned_image, img_a, size,
        destinationIndex=[0, 0, 0],
    )
    aligned_image = sitk.Paste(
        aligned_image, img_b, size,
        destinationIndex=[0, 0, sz],
    )
    return aligned_image


def compute_bounding_box(
    image_size: Tuple[int, int, int],
    coordinates: Tuple[int, int, int],
    volume: Tuple[int, int, int],
) -> Tuple[slice, slice, slice]:
    size_x, size_y, size_z = image_size
    x, y, z = coordinates
    z = size_z - z
    hv_x, hv_y, hv_z = [int(v / 2) for v in volume]

    low_z = max(0, min(z - hv_z, size_z))
    high_z = max(0, min(z + hv_z, size_z))
    low_y = max(0, min(y - hv_y, size_y))
    high_y = max(0, min(y + hv_y, size_y))
    low_x = max(0, min(x - hv_x, size_x))
    high_x = max(0, min(x + hv_x, size_x))

    return (
        slice(low_z, high_z),
        slice(low_y, high_y),
        slice(low_x, high_x),
    )


def match_and_mask_histogram(
    img_arr: np.ndarray, ref_arr: np.ndarray, mask_arr: np.ndarray
) -> np.ndarray:
    img_arr_dim0 = img_arr.shape[0]
    img_arr_2d = img_arr.reshape(-1, img_arr.shape[2])

    matched = match_histograms(img_arr_2d, ref_arr)
    matched = np.ceil(
        matched.reshape(img_arr_dim0, -1, img_arr_2d.shape[1])
    )
    matched[mask_arr <= 0] = 0
    return matched


# --- I/O wrappers ---


def align_images(
    a_file_paths: List[str],
    b_file_paths: List[str],
    target_path: str,
) -> None:
    os.makedirs(target_path, exist_ok=True)

    for i in range(len(a_file_paths)):
        img_a = sitk.ReadImage(a_file_paths[i])
        img_b = sitk.ReadImage(b_file_paths[i])

        try:
            aligned_image = concatenate_images(img_a, img_b)
        except RuntimeError:
            logger.error(
                "Failed to concatenate images: img_a=%s, img_b=%s",
                img_a.GetSize(), img_b.GetSize(),
            )
            continue

        time_period = a_file_paths[i].split("_")[-1].split(".")[0]
        sitk.WriteImage(
            aligned_image,
            os.path.join(target_path, f"{i:04d}_{time_period}.nii"),
        )


def create_train_val_test_split(
    data_paths: List[str], split_perc: float
) -> Tuple[List[str], List[str], List[str]]:
    x_train, x_test = train_test_split(
        data_paths, train_size=split_perc, random_state=42
    )
    x_val, x_test = train_test_split(
        x_test, train_size=0.5, random_state=42
    )
    return x_train, x_val, x_test


def convert_mhd_to_nii(data_path: str) -> None:
    count = 0
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(".mha"):
                filename = os.path.splitext(file)[0]
                file_dir = os.path.join(root, file)
                nii_path = os.path.join(root, filename + ".nii")

                img = sitk.ReadImage(file_dir)
                sitk.WriteImage(img, nii_path)
                count += 1
                logger.info("Converted: %s", file)

    logger.info("Converted %d files", count)


def extract_bounding_box(
    filepath: str,
    coordinates: Tuple[int, int, int] = (0, 0, 0),
    volume: Tuple[int, int, int] = (50, 40, 10),
) -> sitk.Image:
    img = sitk.ReadImage(filepath)
    img = img[::-1, ::-1, :]

    sz, sy, sx = compute_bounding_box(img.GetSize(), coordinates, volume)

    logger.info("Z: [%d : %d]", sz.start, sz.stop)
    logger.info("Y: [%d : %d]", sy.start, sy.stop)
    logger.info("X: [%d : %d]", sx.start, sx.stop)

    return img[sx.start:sx.stop, sy.start:sy.stop, sz.start:sz.stop]


def perform_histogram_matching(
    data_path: str,
    template_path: str,
    mri_type: str = "T2F",
) -> None:
    filetree = parse(data_path, MRI_type=mri_type)

    ref = sitk.ReadImage(template_path)
    ref_arr = sitk.GetArrayFromImage(ref)
    ref_arr = ref_arr.reshape(-1, ref_arr.shape[2])

    for patient, dates in filetree.items():
        logger.info("Patient: %s", patient)

        for date, files in dates.items():
            logger.info("%s", date)
            for file_dir in files:
                logger.info("file: %s", file_dir)
                file_mask_dir = (
                    os.path.dirname(file_dir)
                    + "/alpha_mask_brain_roi.nii.gz"
                )

                img = sitk.ReadImage(file_dir)
                mask = sitk.ReadImage(file_mask_dir)
                img_arr = sitk.GetArrayFromImage(img)
                mask_arr = sitk.GetArrayFromImage(mask)

                matched = match_and_mask_histogram(img_arr, ref_arr, mask_arr)

                new_nii = sitk.GetImageFromArray(matched)
                new_nii.CopyInformation(img)
                new_nii = sitk.Cast(new_nii, sitk.sitkFloat32)
                sitk.WriteImage(new_nii, file_dir)


def create_paired_images(
    data_path: str,
    mri_type: str,
    days: Tuple[Optional[int], Optional[int]] = (None, None),
    store_mask: bool = False,
    split_perc: float = 0.7,
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

    if store_mask:
        masks_dir = os.path.join(new_dir, "masks")
        os.makedirs(masks_dir, exist_ok=True)

    date_format = "%d.%m.%y"
    for patient, dates in filetree.items():
        logger.info("%s", patient)

        dates_list = sorted(
            dates.keys(),
            key=lambda x: datetime.strptime(x, date_format),
        )
        combs = list(itertools.combinations(dates_list, 2))

        for date1, date2 in combs:
            date1_parsed = datetime.strptime(date1, date_format)
            date2_parsed = datetime.strptime(date2, date_format)
            diff = (date2_parsed - date1_parsed).days

            assert diff > 0

            low_day, high_day = days
            if low_day is not None and high_day is not None:
                if diff < low_day or diff > high_day:
                    continue

            img1_path = dates[date1][0]
            img2_path = dates[date2][0]
            logger.info("Date1: %s, Date2: %s", date1, date2)
            logger.info("img1: %s, img2: %s", img1_path, img2_path)
            img1 = sitk.ReadImage(img1_path)
            img2 = sitk.ReadImage(img2_path)

            if store_mask:
                mask1_path = (
                    os.path.dirname(img1_path)
                    + "/alpha_mask_brain_roi.nii.gz"
                )
                mask2_path = (
                    os.path.dirname(img2_path)
                    + "/alpha_mask_brain_roi.nii.gz"
                )
                mask1 = sitk.ReadImage(mask1_path)
                mask2 = sitk.ReadImage(mask2_path)

            weeks = diff // 7
            path_a = os.path.join(a_dir, f"{counter}_{weeks}w.nii")
            path_b = os.path.join(b_dir, f"{counter}_{weeks}w.nii")
            try:
                sitk.WriteImage(img1, path_a)
                sitk.WriteImage(img2, path_b)
            except IndexError:
                continue

            if store_mask:
                path_a_mask = os.path.join(a_dir, f"{counter}_mask_{weeks}w.nii")
                path_b_mask = os.path.join(b_dir, f"{counter}_mask_{weeks}w.nii")
                try:
                    sitk.WriteImage(mask1, path_a_mask)
                    sitk.WriteImage(mask2, path_b_mask)
                except IndexError:
                    continue

            counter += 1

    a_paths = get_file_paths(a_dir)
    b_paths = get_file_paths(b_dir)
    align_images(a_paths, b_paths, ab_dir)

    if store_mask:
        a_mask_paths = get_file_paths(a_dir, masks=True)
        b_mask_paths = get_file_paths(b_dir, masks=True)
        align_images(a_mask_paths, b_mask_paths, masks_dir)

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


def delete_by_type(data_path: str, mri_type: str) -> None:
    count = 0
    filetree = parse(data_path, MRI_type=mri_type)
    for patient, dates in filetree.items():
        for date, files in dates.items():
            for file in files:
                os.remove(file)
                count += 1
    logger.info("Deleted %d files of type %s.", count, mri_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("rootdir", help="Directory of Data")
    args = parser.parse_args()
    create_paired_images(args.rootdir, "T2F", store_mask=False)
