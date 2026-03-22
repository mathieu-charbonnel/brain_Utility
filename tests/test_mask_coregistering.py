import numpy as np
import SimpleITK as sitk

from brain_utility.coregister import build_registration_method
from brain_utility.mask_coregistering import register_image_pair


class TestRegisterImagePair:
    def test_returns_two_images_with_correct_sizes(self) -> None:
        arr = np.random.rand(64, 64, 64).astype(np.float32) * 1000
        seg_arr = np.random.rand(64, 64, 64).astype(np.float32)

        fixed = sitk.GetImageFromArray(arr)
        fixed.SetSpacing([1.0, 1.0, 1.0])
        fixed.SetOrigin([0.0, 0.0, 0.0])

        moving = sitk.GetImageFromArray(arr)
        moving.SetSpacing([1.0, 1.0, 1.0])
        moving.SetOrigin([0.0, 0.0, 0.0])

        fixed_seg = sitk.GetImageFromArray(seg_arr)
        fixed_seg.SetSpacing([1.0, 1.0, 1.0])
        fixed_seg.SetOrigin([0.0, 0.0, 0.0])

        moving_seg = sitk.GetImageFromArray(seg_arr)
        moving_seg.SetSpacing([1.0, 1.0, 1.0])
        moving_seg.SetOrigin([0.0, 0.0, 0.0])

        rmethod = build_registration_method()
        result_img, result_seg = register_image_pair(
            fixed, moving, fixed_seg, moving_seg, rmethod
        )

        assert result_img.GetSize() == fixed.GetSize()
        assert result_seg.GetSize() == fixed_seg.GetSize()
        assert result_img.GetPixelID() == sitk.sitkInt16
        assert result_seg.GetPixelID() == sitk.sitkInt16
