import os
from typing import Dict, List, Optional, Tuple


DATE_FORMAT = "%d.%m.%y"

MRI_EXTENSIONS = (".nii", ".nii.gz")

T1_KEYWORDS = ("T1", "t1")
T2F_KEYWORDS = ("T2F", "t2f", "FLAIR", "flair", "T2_FLAIR", "t2_flair")
SEGMENTATION_KEYWORDS = ("seg", "Seg", "segmentation", "label", "mask")


def T1_exists(filepath: str) -> bool:
    basename = os.path.basename(filepath)
    return any(kw in basename for kw in T1_KEYWORDS)


def T2F_exists(filepath: str) -> bool:
    basename = os.path.basename(filepath)
    return any(kw in basename for kw in T2F_KEYWORDS)


def segmentation_exists(filepath: str) -> bool:
    basename = os.path.basename(filepath)
    return any(kw in basename for kw in SEGMENTATION_KEYWORDS)


def _is_nifti(filename: str) -> bool:
    return filename.endswith(".nii.gz") or filename.endswith(".nii")


def get_file_paths(
    folder: str, masks: bool = False
) -> List[str]:
    file_paths: List[str] = []
    for root, _dirs, filenames in os.walk(folder):
        for filename in sorted(filenames):
            if not _is_nifti(filename):
                continue
            if not masks and "_mask" in filename:
                continue
            if masks and "_mask" not in filename:
                continue
            file_paths.append(os.path.join(root, filename))
    return file_paths


def parse(
    data_path: str, MRI_type: Optional[str] = None
) -> Dict[str, Dict[str, List[str]]]:
    filetree: Dict[str, Dict[str, List[str]]] = {}

    for patient_dir in sorted(os.listdir(data_path)):
        patient_path = os.path.join(data_path, patient_dir)
        if not os.path.isdir(patient_path):
            continue

        dates: Dict[str, List[str]] = {}
        for date_dir in sorted(os.listdir(patient_path)):
            date_path = os.path.join(patient_path, date_dir)
            if not os.path.isdir(date_path):
                continue

            files: List[str] = []
            for root, _dirs, filenames in os.walk(date_path):
                for filename in sorted(filenames):
                    if not _is_nifti(filename):
                        continue
                    filepath = os.path.join(root, filename)
                    if MRI_type is not None:
                        if MRI_type == "T1" and not T1_exists(filepath):
                            continue
                        if MRI_type == "T2F" and not T2F_exists(filepath):
                            continue
                    files.append(filepath)

            if files:
                dates[date_dir] = files

        if dates:
            filetree[patient_dir] = dates

    return filetree


def parse_tumor_masking(
    data_path: str,
) -> Dict[str, Dict[str, List[str]]]:
    filetree: Dict[str, Dict[str, List[str]]] = {}

    for patient_dir in sorted(os.listdir(data_path)):
        patient_path = os.path.join(data_path, patient_dir)
        if not os.path.isdir(patient_path):
            continue

        dates: Dict[str, List[str]] = {}
        for date_dir in sorted(os.listdir(patient_path)):
            date_path = os.path.join(patient_path, date_dir)
            if not os.path.isdir(date_path):
                continue

            files: List[str] = []
            for root, _dirs, filenames in os.walk(date_path):
                for filename in sorted(filenames):
                    if not _is_nifti(filename):
                        continue
                    filepath = os.path.join(root, filename)
                    if T1_exists(filepath) or segmentation_exists(filepath):
                        files.append(filepath)

            if files:
                dates[date_dir] = files

        if dates:
            filetree[patient_dir] = dates

    return filetree


def separate_rA_rB_fB(
    file_paths: List[str],
) -> Tuple[List[str], List[str], List[str]]:
    real_a: List[str] = []
    real_b: List[str] = []
    fake_b: List[str] = []

    for path in sorted(file_paths):
        basename = os.path.basename(path)
        if "real_A" in basename:
            real_a.append(path)
        elif "real_B" in basename:
            real_b.append(path)
        elif "fake_B" in basename:
            fake_b.append(path)

    return real_a, real_b, fake_b
