import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from brain_utility.dm_preprocessing import (
    compute_difference_map,
    create_paired_dm_images,
)
from brain_utility.preprocessing import create_train_val_test_split


class TestComputeDifferenceMap:
    def test_basic_subtraction(self) -> None:
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([4.0, 6.0, 8.0])
        result = compute_difference_map(arr1, arr2)
        np.testing.assert_array_equal(result, np.array([3.0, 4.0, 5.0]))

    def test_identical_arrays_give_zeros(self) -> None:
        arr = np.array([10.0, 20.0, 30.0])
        result = compute_difference_map(arr, arr)
        np.testing.assert_array_equal(result, np.zeros(3))

    def test_preserves_shape(self) -> None:
        arr1 = np.ones((4, 4, 4))
        arr2 = np.ones((4, 4, 4)) * 2
        result = compute_difference_map(arr1, arr2)
        assert result.shape == (4, 4, 4)


class TestDmPreprocessing:
    def test_create_paired_dm_images_rejects_invalid_type(self) -> None:
        with pytest.raises(ValueError, match="Invalid MRI type"):
            create_paired_dm_images("/some/path", "INVALID")

    @patch("brain_utility.dm_preprocessing.shutil")
    @patch("brain_utility.dm_preprocessing.align_images")
    @patch("brain_utility.dm_preprocessing.os.makedirs")
    def test_create_paired_dm_images_accepts_t1(
        self, mock_makedirs: MagicMock, mock_align: MagicMock,
        mock_shutil: MagicMock,
    ) -> None:
        with patch("brain_utility.dm_preprocessing.parse", return_value={}):
            with patch(
                "brain_utility.dm_preprocessing.get_file_paths",
                return_value=["a.nii", "b.nii", "c.nii", "d.nii"],
            ):
                create_paired_dm_images("/some/path", "T1")

    @patch("brain_utility.dm_preprocessing.shutil")
    @patch("brain_utility.dm_preprocessing.align_images")
    @patch("brain_utility.dm_preprocessing.os.makedirs")
    def test_create_paired_dm_images_accepts_t2f(
        self, mock_makedirs: MagicMock, mock_align: MagicMock,
        mock_shutil: MagicMock,
    ) -> None:
        with patch("brain_utility.dm_preprocessing.parse", return_value={}):
            with patch(
                "brain_utility.dm_preprocessing.get_file_paths",
                return_value=["a.nii", "b.nii", "c.nii", "d.nii"],
            ):
                create_paired_dm_images("/some/path", "T2F")

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
