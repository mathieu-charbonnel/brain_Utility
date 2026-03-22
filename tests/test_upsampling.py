import os
from unittest.mock import MagicMock, patch

import pytest
import SimpleITK as sitk

from brain_utility.upsampling import get_nifti_file_paths, resample_image, resample_images


class TestResampleImage:
    def test_output_has_target_size(self) -> None:
        img = sitk.Image(128, 128, 128, sitk.sitkFloat32)
        img.SetSpacing([1.0, 1.0, 1.0])
        result = resample_image(img, [64, 64, 64])
        assert result.GetSize() == (64, 64, 64)

    def test_preserves_origin(self) -> None:
        img = sitk.Image(64, 64, 64, sitk.sitkFloat32)
        img.SetOrigin((10.0, 20.0, 30.0))
        img.SetSpacing([1.0, 1.0, 1.0])
        result = resample_image(img, [32, 32, 32])
        assert result.GetOrigin() == (10.0, 20.0, 30.0)

    def test_spacing_scales_correctly(self) -> None:
        img = sitk.Image(128, 128, 128, sitk.sitkFloat32)
        img.SetSpacing([1.0, 1.0, 1.0])
        result = resample_image(img, [64, 64, 64])
        assert result.GetSpacing() == pytest.approx((2.0, 2.0, 2.0))


class TestGetNiftiFilePaths:
    @patch("brain_utility.upsampling.os.walk")
    def test_returns_nii_files(self, mock_walk: MagicMock) -> None:
        mock_walk.return_value = [
            ("/data", [], ["scan1.nii", "scan2.nii.gz", "readme.txt"]),
        ]
        result = get_nifti_file_paths("/data")
        assert len(result) == 2
        assert any("scan1.nii" in p for p in result)
        assert any("scan2.nii.gz" in p for p in result)

    @patch("brain_utility.upsampling.os.walk")
    def test_excludes_masks_by_default(self, mock_walk: MagicMock) -> None:
        mock_walk.return_value = [
            ("/data", [], ["scan.nii", "scan_mask.nii"]),
        ]
        result = get_nifti_file_paths("/data", masks=False)
        assert len(result) == 1
        assert any("scan.nii" in p for p in result)

    @patch("brain_utility.upsampling.os.walk")
    def test_returns_only_masks_when_requested(
        self, mock_walk: MagicMock
    ) -> None:
        mock_walk.return_value = [
            ("/data", [], ["scan.nii", "scan_mask.nii"]),
        ]
        result = get_nifti_file_paths("/data", masks=True)
        assert len(result) == 1
        assert any("_mask" in p for p in result)

    @patch("brain_utility.upsampling.os.walk")
    def test_returns_empty_for_no_nii_files(
        self, mock_walk: MagicMock
    ) -> None:
        mock_walk.return_value = [
            ("/data", [], ["readme.txt", "data.csv"]),
        ]
        result = get_nifti_file_paths("/data")
        assert len(result) == 0

    @patch("brain_utility.upsampling.os.walk")
    def test_returns_sorted_files(self, mock_walk: MagicMock) -> None:
        mock_walk.return_value = [
            ("/data", [], ["z_scan.nii", "a_scan.nii", "m_scan.nii"]),
        ]
        result = get_nifti_file_paths("/data")
        filenames = [os.path.basename(p) for p in result]
        assert filenames == ["a_scan.nii", "m_scan.nii", "z_scan.nii"]


class TestResampleImages:
    @patch("brain_utility.upsampling.get_nifti_file_paths")
    @patch("brain_utility.upsampling.sitk")
    def test_resample_creates_reference_image(
        self, mock_sitk: MagicMock, mock_get_paths: MagicMock
    ) -> None:
        mock_get_paths.return_value = ["/data/scan.nii"]

        mock_img = MagicMock()
        mock_img.GetPixelIDValue.return_value = 8
        mock_img.GetOrigin.return_value = (0.0, 0.0, 0.0)
        mock_img.GetDirection.return_value = (1, 0, 0, 0, 1, 0, 0, 0, 1)
        mock_img.GetSize.return_value = (128, 128, 128)
        mock_img.GetSpacing.return_value = (1.0, 1.0, 1.0)
        mock_sitk.ReadImage.return_value = mock_img

        mock_ref = MagicMock()
        mock_sitk.Image.return_value = mock_ref
        mock_resampled = MagicMock()
        mock_sitk.Resample.return_value = mock_resampled

        resample_images("/data", [64, 64, 64])

        mock_sitk.Image.assert_called_once_with([64, 64, 64], 8)
        mock_ref.SetOrigin.assert_called_once_with((0.0, 0.0, 0.0))
        mock_ref.SetDirection.assert_called_once_with(
            (1, 0, 0, 0, 1, 0, 0, 0, 1)
        )
        mock_sitk.Resample.assert_called_once_with(mock_img, mock_ref)
        mock_sitk.WriteImage.assert_called_once_with(
            mock_resampled, "/data/scan.nii"
        )

    @patch("brain_utility.upsampling.get_nifti_file_paths")
    @patch("brain_utility.upsampling.sitk")
    def test_resample_computes_correct_spacing(
        self, mock_sitk: MagicMock, mock_get_paths: MagicMock
    ) -> None:
        mock_get_paths.return_value = ["/data/scan.nii"]

        mock_img = MagicMock()
        mock_img.GetPixelIDValue.return_value = 8
        mock_img.GetOrigin.return_value = (0.0, 0.0, 0.0)
        mock_img.GetDirection.return_value = (1, 0, 0, 0, 1, 0, 0, 0, 1)
        mock_img.GetSize.return_value = (128, 128, 128)
        mock_img.GetSpacing.return_value = (1.0, 1.0, 1.0)
        mock_sitk.ReadImage.return_value = mock_img

        mock_ref = MagicMock()
        mock_sitk.Image.return_value = mock_ref

        resample_images("/data", [64, 64, 64])

        expected_spacing = [128 * 1.0 / 64] * 3
        mock_ref.SetSpacing.assert_called_once()
        actual_spacing = mock_ref.SetSpacing.call_args[0][0]
        assert actual_spacing == pytest.approx(expected_spacing)
