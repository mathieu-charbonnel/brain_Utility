import os
from typing import List

import SimpleITK as sitk


def get_nifti_file_paths(
    folder: str, masks: bool = False
) -> List[str]:
    image_file_paths: List[str] = []
    for root, dirs, filenames in os.walk(folder):
        filenames = sorted(filenames)
        for filename in filenames:
            input_path = os.path.abspath(root)
            file_path = os.path.join(input_path, filename)
            if filename.endswith(".nii") or filename.endswith(".nii.gz"):
                if not masks and "_mask" in filename:
                    continue
                elif masks and "_mask" not in filename:
                    continue
                image_file_paths.append(file_path)
    return image_file_paths


# --- Pure functions ---


def resample_image(
    img: sitk.Image, target_size: List[int]
) -> sitk.Image:
    reference_image = sitk.Image(target_size, img.GetPixelIDValue())
    reference_image.SetOrigin(img.GetOrigin())
    reference_image.SetDirection(img.GetDirection())
    reference_image.SetSpacing(
        [
            sz * spc / nsz
            for nsz, sz, spc in zip(
                target_size, img.GetSize(), img.GetSpacing()
            )
        ]
    )
    return sitk.Resample(img, reference_image)


# --- I/O wrappers ---


def resample_images(
    data_dir: str, new_size: List[int]
) -> None:
    for file_path in get_nifti_file_paths(data_dir):
        img = sitk.ReadImage(file_path)
        resampled = resample_image(img, new_size)
        sitk.WriteImage(resampled, file_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Directory containing NIfTI files")
    parser.add_argument(
        "--size",
        type=int,
        nargs=3,
        default=[64, 64, 64],
        help="Target size as three integers (default: 64 64 64)",
    )
    args = parser.parse_args()
    resample_images(args.data_dir, args.size)
