import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from brain_utility.preprocessing import (
    compute_bounding_box,
    concatenate_images,
    create_paired_images,
    create_train_val_test_split,
    match_and_mask_histogram,
)


class TestComputeBoundingBox:
    def test_centered_box(self) -> None:
        sz, sy, sx = compute_bounding_box(
            image_size=(128, 128, 128),
            coordinates=(64, 64, 64),
            volume=(20, 20, 20),
        )
        assert sx == slice(54, 74)
        assert sy == slice(54, 74)
        assert sz == slice(54, 74)

    def test_clamps_to_image_bounds(self) -> None:
        sz, sy, sx = compute_bounding_box(
            image_size=(128, 128, 128),
            coordinates=(0, 0, 128),
            volume=(20, 20, 20),
        )
        assert sx.start >= 0
        assert sy.start >= 0
        assert sz.start >= 0
        assert sx.stop <= 128
        assert sy.stop <= 128
        assert sz.stop <= 128

    def test_z_flipped(self) -> None:
        sz, sy, sx = compute_bounding_box(
            image_size=(100, 100, 100),
            coordinates=(50, 50, 10),
            volume=(10, 10, 10),
        )
        # z should be flipped: size_z - z = 100 - 10 = 90
        assert sz == slice(85, 95)


class TestMatchAndMaskHistogram:
    def test_zeros_out_masked_regions(self) -> None:
        img_arr = np.ones((4, 4, 4), dtype=np.float64) * 100
        ref_arr = np.ones((16, 4), dtype=np.float64) * 100
        mask_arr = np.zeros((4, 4, 4), dtype=np.float64)
        mask_arr[0:2, :, :] = 1  # only first 2 slices are unmasked

        result = match_and_mask_histogram(img_arr, ref_arr, mask_arr)
        # masked region should be zero
        assert np.all(result[2:, :, :] == 0)
        assert result.shape == img_arr.shape


class TestPreprocessing:
    def test_create_paired_images_rejects_invalid_type(self) -> None:
        with pytest.raises(ValueError, match="Invalid MRI type"):
            create_paired_images("/some/path", "INVALID")

    @patch("brain_utility.preprocessing.shutil")
    @patch("brain_utility.preprocessing.align_images")
    @patch("brain_utility.preprocessing.os.makedirs")
    def test_create_paired_images_accepts_t1(
        self, mock_makedirs: MagicMock, mock_align: MagicMock,
        mock_shutil: MagicMock,
    ) -> None:
        with patch("brain_utility.preprocessing.parse", return_value={}):
            with patch(
                "brain_utility.preprocessing.get_file_paths",
                return_value=["a.nii", "b.nii", "c.nii", "d.nii"],
            ):
                create_paired_images("/some/path", "T1")

    @patch("brain_utility.preprocessing.shutil")
    @patch("brain_utility.preprocessing.align_images")
    @patch("brain_utility.preprocessing.os.makedirs")
    def test_create_paired_images_accepts_t2f(
        self, mock_makedirs: MagicMock, mock_align: MagicMock,
        mock_shutil: MagicMock,
    ) -> None:
        with patch("brain_utility.preprocessing.parse", return_value={}):
            with patch(
                "brain_utility.preprocessing.get_file_paths",
                return_value=["a.nii", "b.nii", "c.nii", "d.nii"],
            ):
                create_paired_images("/some/path", "T2F")

    def test_create_train_val_test_split_sizes(self) -> None:
        data = list(range(100))
        train, val, test = create_train_val_test_split(data, split_perc=0.7)
        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15

    def test_create_train_val_test_split_no_overlap(self) -> None:
        data = list(range(100))
        train, val, test = create_train_val_test_split(data, split_perc=0.7)
        all_items = set(train) | set(val) | set(test)
        assert len(all_items) == 100

    def test_create_train_val_test_split_deterministic(self) -> None:
        data = list(range(100))
        train1, val1, test1 = create_train_val_test_split(data, split_perc=0.7)
        train2, val2, test2 = create_train_val_test_split(data, split_perc=0.7)
        assert train1 == train2
        assert val1 == val2
        assert test1 == test2

    def test_create_train_val_test_split_different_ratios(self) -> None:
        data = list(range(100))
        train, val, test = create_train_val_test_split(data, split_perc=0.8)
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10
