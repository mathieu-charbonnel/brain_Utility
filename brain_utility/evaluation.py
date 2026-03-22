import argparse
import logging
from typing import List, Optional, Tuple

import numpy as np
import SimpleITK as sitk
from skimage.metrics import structural_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error

from brain_utility.utils.parser import get_file_paths, separate_rA_rB_fB

logger = logging.getLogger(__name__)

MAX_PIXEL = 100


# --- Pure functions ---


def denormalize_array(arr: np.ndarray) -> np.ndarray:
    return (arr + 1) / 2.0 * 255.0


def _threshold_mask(arr: np.ndarray, threshold: float, below: bool) -> np.ndarray:
    if below:
        return np.array((arr < threshold), dtype=np.float32)
    return np.array((arr > threshold), dtype=np.float32)


def mae_under_threshold(
    img1: np.ndarray, img2: np.ndarray, threshold: float = 100.0
) -> float:
    weight = _threshold_mask(img1, threshold, below=True) * _threshold_mask(img2, threshold, below=True)
    weighted_diff = weight * np.abs(img1 - img2)
    return float(np.sum(weighted_diff) / np.sum(weight))


def mae_over_threshold(
    img1: np.ndarray, img2: np.ndarray, threshold: float = 100.0
) -> float:
    mask_1 = _threshold_mask(img1, threshold, below=False)
    mask_2 = _threshold_mask(img2, threshold, below=False)
    mask = np.logical_or(mask_1, mask_2).astype(np.float32)
    weighted_diff = mask * np.abs(img1 - img2)
    return float(np.sum(weighted_diff) / np.sum(mask))


def compute_mae(data_fake: np.ndarray, data_real: np.ndarray) -> float:
    return float(mean_absolute_error(y_true=data_real, y_pred=data_fake))


def compute_mse(data_fake: np.ndarray, data_real: np.ndarray) -> float:
    return float(mean_squared_error(y_true=data_real, y_pred=data_fake))


def compute_psnr(data_fake: np.ndarray, data_real: np.ndarray) -> float:
    mse = compute_mse(data_fake, data_real)
    return float(20 * np.log10(MAX_PIXEL) - 10 * np.log10(mse))


def compute_ssim(data_fake: np.ndarray, data_real: np.ndarray) -> float:
    cumulative_ssim = 0.0
    for i in range(data_fake.shape[0]):
        cumulative_ssim += structural_similarity(data_real[i], data_fake[i])
    return cumulative_ssim / data_fake.shape[0]


def reconstruction_score(
    data_fake: np.ndarray, data_input: np.ndarray
) -> float:
    num_samples = data_fake.shape[0]
    score = 0

    for i in range(num_samples):
        distance_matrix = np.zeros(num_samples)
        distance_corresponding = mae_under_threshold(
            data_input[i], data_fake[i]
        )

        for j in range(num_samples):
            distance_matrix[j] = mae_under_threshold(
                data_input[j], data_fake[i]
            )

        k = 2
        idx = np.argpartition(distance_matrix, k)[:k]
        idx = idx[np.argsort(distance_matrix[idx])]

        if (
            np.abs(distance_matrix[idx[0]] - distance_corresponding) < 0.01
            or np.abs(distance_matrix[idx[1]] - distance_corresponding) < 0.1
        ):
            score += 1

    return score / num_samples


def evolution_score(
    data_fake: np.ndarray, data_real: np.ndarray
) -> float:
    num_samples = data_fake.shape[0]
    score = 0

    for i in range(num_samples):
        distance_matrix = np.zeros(num_samples)
        distance_corresponding = mae_over_threshold(
            data_real[i], data_fake[i]
        )

        for j in range(num_samples):
            distance_matrix[j] = mae_over_threshold(
                data_real[j], data_fake[i]
            )

        k = 2
        idx = np.argpartition(distance_matrix, k)[:k]
        idx = idx[np.argsort(distance_matrix[idx])]

        if (
            np.abs(distance_matrix[idx[0]] - distance_corresponding) < 0.1
            or np.abs(distance_matrix[idx[1]] - distance_corresponding) < 0.1
        ):
            score += 1

    return score / num_samples


# --- I/O wrappers ---


def load_data(
    data_path: str,
    image_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    logger.info("Loading from: %s", data_path)
    file_paths = get_file_paths(data_path)

    if len(file_paths) <= 0:
        logger.warning("No image files found")
        return np.array([]), np.array([]), np.array([])

    real_a, real_b, fake_b = separate_rA_rB_fB(file_paths)

    sample_count = len(real_a)
    flat_size = image_size ** 3

    dataset_fake = np.ndarray(
        shape=(sample_count, flat_size), dtype=np.float32
    )
    dataset_real = np.ndarray(
        shape=(sample_count, flat_size), dtype=np.float32
    )
    dataset_input = np.ndarray(
        shape=(sample_count, flat_size), dtype=np.float32
    )

    for i in range(sample_count):
        dataset_fake[i] = sitk.GetArrayFromImage(
            sitk.ReadImage(fake_b[i])
        ).reshape(-1)
        dataset_real[i] = sitk.GetArrayFromImage(
            sitk.ReadImage(real_b[i])
        ).reshape(-1)
        dataset_input[i] = sitk.GetArrayFromImage(
            sitk.ReadImage(real_a[i])
        ).reshape(-1)

    dataset_fake = denormalize_array(dataset_fake)
    dataset_real = denormalize_array(dataset_real)
    dataset_input = denormalize_array(dataset_input)
    return dataset_fake, dataset_real, dataset_input


def main(args: argparse.Namespace) -> None:
    logger.info("Evaluating metrics:")

    data_path = args.rootdir
    data_fake, data_real, data_input = load_data(data_path)

    logger.info("MAE: %s", compute_mae(data_fake, data_real))
    logger.info("MSE: %s", compute_mse(data_fake, data_real))
    logger.info("SSIM: %s", compute_ssim(data_fake, data_real))
    logger.info("Reconstruction Score: %s", reconstruction_score(data_fake, data_input))
    logger.info("Evolution Score: %s", evolution_score(data_fake, data_real))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("rootdir", help="Directory of Data")
    main(parser.parse_args())
