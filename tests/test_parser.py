import os
import tempfile

from brain_utility.utils.parser import (
    T1_exists,
    T2F_exists,
    get_file_paths,
    segmentation_exists,
    separate_rA_rB_fB,
)


class TestT1Exists:
    def test_detects_t1(self) -> None:
        assert T1_exists("/data/patient/T1_scan.nii")

    def test_rejects_non_t1(self) -> None:
        assert not T1_exists("/data/patient/FLAIR_scan.nii")


class TestT2FExists:
    def test_detects_flair(self) -> None:
        assert T2F_exists("/data/patient/T2F_scan.nii")
        assert T2F_exists("/data/patient/FLAIR_scan.nii")

    def test_rejects_non_flair(self) -> None:
        assert not T2F_exists("/data/patient/T1_scan.nii")


class TestSegmentationExists:
    def test_detects_segmentation(self) -> None:
        assert segmentation_exists("/data/patient/seg_brain.nii")
        assert segmentation_exists("/data/patient/mask_tumor.nii")

    def test_rejects_non_segmentation(self) -> None:
        assert not segmentation_exists("/data/patient/T1_scan.nii")


class TestGetFilePaths:
    def test_finds_nii_files(self, tmp_path: os.PathLike) -> None:
        (tmp_path / "scan.nii").write_text("")
        (tmp_path / "scan.nii.gz").write_text("")
        (tmp_path / "readme.txt").write_text("")

        result = get_file_paths(str(tmp_path))
        assert len(result) == 2

    def test_excludes_masks_by_default(self, tmp_path: os.PathLike) -> None:
        (tmp_path / "scan.nii").write_text("")
        (tmp_path / "scan_mask.nii").write_text("")

        result = get_file_paths(str(tmp_path), masks=False)
        assert len(result) == 1
        assert os.path.basename(result[0]) == "scan.nii"

    def test_returns_only_masks(self, tmp_path: os.PathLike) -> None:
        (tmp_path / "scan.nii").write_text("")
        (tmp_path / "scan_mask.nii").write_text("")

        result = get_file_paths(str(tmp_path), masks=True)
        assert len(result) == 1
        assert "_mask" in result[0]


class TestSeparateRARBFB:
    def test_separates_correctly(self) -> None:
        paths = [
            "/data/001_real_A.nii",
            "/data/001_real_B.nii",
            "/data/001_fake_B.nii",
            "/data/002_real_A.nii",
            "/data/002_real_B.nii",
            "/data/002_fake_B.nii",
        ]
        real_a, real_b, fake_b = separate_rA_rB_fB(paths)
        assert len(real_a) == 2
        assert len(real_b) == 2
        assert len(fake_b) == 2

    def test_empty_input(self) -> None:
        real_a, real_b, fake_b = separate_rA_rB_fB([])
        assert real_a == []
        assert real_b == []
        assert fake_b == []
