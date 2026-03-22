import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from brain_utility.rescaling import normalize_array, rescale


class TestNormalizeArray:
    def test_maps_min_to_negative_one(self) -> None:
        arr = np.array([-1000.0])
        result = normalize_array(arr, pixel_min=-1000.0, pixel_max=1989.0)
        assert result[0] == pytest.approx(-1.0)

    def test_maps_max_to_positive_one(self) -> None:
        arr = np.array([1989.0])
        result = normalize_array(arr, pixel_min=-1000.0, pixel_max=1989.0)
        assert result[0] == pytest.approx(1.0)

    def test_maps_midpoint_to_zero(self) -> None:
        arr = np.array([494.5])
        result = normalize_array(arr, pixel_min=-1000.0, pixel_max=1989.0)
        assert result[0] == pytest.approx(0.0)

    def test_symmetric_range(self) -> None:
        arr = np.array([0.0, 50.0, 100.0, 200.0])
        result = normalize_array(arr, pixel_min=0.0, pixel_max=200.0)
        expected = (arr - 100.0) / 100.0
        np.testing.assert_array_almost_equal(result, expected)

    def test_preserves_shape(self) -> None:
        arr = np.ones((3, 4, 5))
        result = normalize_array(arr, pixel_min=0.0, pixel_max=2.0)
        assert result.shape == (3, 4, 5)


class TestRescale:
    @patch("brain_utility.rescaling.sitk")
    @patch("brain_utility.rescaling.os.walk")
    def test_default_params(
        self, mock_walk: MagicMock, mock_sitk: MagicMock
    ) -> None:
        mock_walk.return_value = [("/data", [], ["scan.nii"])]

        input_array = np.array([0.0, 494.5, 1989.0, -1000.0])
        mock_sitk.ReadImage.return_value = MagicMock()
        mock_sitk.GetArrayFromImage.return_value = input_array.copy()

        rescale("/data")

        written_array = mock_sitk.GetImageFromArray.call_args[0][0]
        half_range = (1989.0 - (-1000.0)) / 2.0
        midpoint = (1989.0 + (-1000.0)) / 2.0
        expected = (input_array - midpoint) / half_range
        np.testing.assert_array_almost_equal(written_array, expected)

    @patch("brain_utility.rescaling.sitk")
    @patch("brain_utility.rescaling.os.walk")
    def test_custom_params(
        self, mock_walk: MagicMock, mock_sitk: MagicMock
    ) -> None:
        mock_walk.return_value = [("/data", [], ["scan.nii"])]

        input_array = np.array([0.0, 50.0, 100.0, 200.0])
        mock_sitk.ReadImage.return_value = MagicMock()
        mock_sitk.GetArrayFromImage.return_value = input_array.copy()

        rescale("/data", pixel_max=200.0, pixel_min=0.0)

        written_array = mock_sitk.GetImageFromArray.call_args[0][0]
        expected = (input_array - 100.0) / 100.0
        np.testing.assert_array_almost_equal(written_array, expected)

    @patch("brain_utility.rescaling.sitk")
    @patch("brain_utility.rescaling.os.walk")
    def test_normalization_range(
        self, mock_walk: MagicMock, mock_sitk: MagicMock
    ) -> None:
        mock_walk.return_value = [("/data", [], ["scan.nii"])]

        input_array = np.array([-1000.0, 1989.0])
        mock_sitk.ReadImage.return_value = MagicMock()
        mock_sitk.GetArrayFromImage.return_value = input_array.copy()

        rescale("/data")

        written_array = mock_sitk.GetImageFromArray.call_args[0][0]
        assert written_array[0] == pytest.approx(-1.0)
        assert written_array[1] == pytest.approx(1.0)

    @patch("brain_utility.rescaling.sitk")
    @patch("brain_utility.rescaling.os.walk")
    def test_multiple_files(
        self, mock_walk: MagicMock, mock_sitk: MagicMock
    ) -> None:
        mock_walk.return_value = [("/data", [], ["scan1.nii", "scan2.nii"])]

        input_array = np.array([0.0, 100.0])
        mock_sitk.ReadImage.return_value = MagicMock()
        mock_sitk.GetArrayFromImage.return_value = input_array.copy()

        rescale("/data")

        assert mock_sitk.ReadImage.call_count == 2
        assert mock_sitk.WriteImage.call_count == 2
