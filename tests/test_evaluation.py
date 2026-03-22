import numpy as np
import pytest
from unittest.mock import patch

from brain_utility.evaluation import (
    MAX_PIXEL,
    compute_mae,
    compute_mse,
    compute_psnr,
    compute_ssim,
    denormalize_array,
    mae_over_threshold,
    mae_under_threshold,
    reconstruction_score,
    evolution_score,
)


class TestDenormalizeArray:
    def test_maps_minus_one_to_zero(self) -> None:
        arr = np.array([-1.0])
        result = denormalize_array(arr)
        assert result[0] == pytest.approx(0.0)

    def test_maps_one_to_255(self) -> None:
        arr = np.array([1.0])
        result = denormalize_array(arr)
        assert result[0] == pytest.approx(255.0)

    def test_maps_zero_to_127_5(self) -> None:
        arr = np.array([0.0])
        result = denormalize_array(arr)
        assert result[0] == pytest.approx(127.5)

    def test_preserves_shape(self) -> None:
        arr = np.zeros((5, 10))
        result = denormalize_array(arr)
        assert result.shape == (5, 10)


class TestEvaluation:
    def test_max_pixel_constant(self) -> None:
        assert MAX_PIXEL == 100

    def test_compute_mae_identical(self) -> None:
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert compute_mae(data, data) == pytest.approx(0.0)

    def test_compute_mae_known_values(self) -> None:
        data_fake = np.array([[1.0, 2.0, 3.0]])
        data_real = np.array([[4.0, 5.0, 6.0]])
        assert compute_mae(data_fake, data_real) == pytest.approx(3.0)

    def test_compute_mse_identical(self) -> None:
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert compute_mse(data, data) == pytest.approx(0.0)

    def test_compute_mse_known_values(self) -> None:
        data_fake = np.array([[1.0, 2.0, 3.0]])
        data_real = np.array([[4.0, 5.0, 6.0]])
        assert compute_mse(data_fake, data_real) == pytest.approx(9.0)

    def test_compute_psnr_known_values(self) -> None:
        data_fake = np.array([[10.0, 20.0, 30.0]])
        data_real = np.array([[11.0, 21.0, 31.0]])
        mse = compute_mse(data_fake, data_real)
        expected_psnr = 20 * np.log10(100) - 10 * np.log10(mse)
        assert compute_psnr(data_fake, data_real) == pytest.approx(expected_psnr)

    def test_mae_under_threshold_all_below(self) -> None:
        img1 = np.array([10.0, 20.0, 30.0])
        img2 = np.array([15.0, 25.0, 35.0])
        result = mae_under_threshold(img1, img2, threshold=100.0)
        expected = np.mean(np.abs(img1 - img2))
        assert result == pytest.approx(expected)

    def test_mae_under_threshold_all_above(self) -> None:
        img1 = np.array([200.0, 300.0, 400.0])
        img2 = np.array([250.0, 350.0, 450.0])
        result = mae_under_threshold(img1, img2, threshold=100.0)
        assert result == pytest.approx(0.0) or np.isnan(result)

    def test_mae_over_threshold_all_above(self) -> None:
        img1 = np.array([200.0, 300.0, 400.0])
        img2 = np.array([250.0, 350.0, 450.0])
        result = mae_over_threshold(img1, img2, threshold=100.0)
        expected = np.mean(np.abs(img1 - img2))
        assert result == pytest.approx(expected)

    def test_mae_over_threshold_all_below(self) -> None:
        img1 = np.array([10.0, 20.0, 30.0])
        img2 = np.array([15.0, 25.0, 35.0])
        result = mae_over_threshold(img1, img2, threshold=100.0)
        assert result == pytest.approx(0.0) or np.isnan(result)

    def test_reconstruction_score_perfect_match(self) -> None:
        data = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0]])
        result = reconstruction_score(data, data)
        assert 0.0 <= result <= 1.0

    def test_evolution_score_returns_bounded_value(self) -> None:
        data = np.array([[110.0, 120.0, 130.0], [140.0, 150.0, 160.0], [170.0, 180.0, 190.0]])
        result = evolution_score(data, data)
        assert 0.0 <= result <= 1.0

    def test_compute_ssim_identical_images(self) -> None:
        data = np.random.rand(3, 100).astype(np.float32)
        with patch(
            "brain_utility.evaluation.structural_similarity",
            return_value=1.0,
        ):
            result = compute_ssim(data, data)
        assert result == pytest.approx(1.0)
