import numpy as np
import SimpleITK as sitk

from brain_utility.coregister import build_registration_method, register_image


class TestBuildRegistrationMethod:
    def test_returns_registration_method(self) -> None:
        rmethod = build_registration_method()
        assert isinstance(rmethod, sitk.ImageRegistrationMethod)


class TestRegisterImage:
    def test_returns_image_with_correct_size(self) -> None:
        arr = np.random.rand(64, 64, 64).astype(np.float32) * 1000
        fixed = sitk.GetImageFromArray(arr)
        fixed.SetSpacing([1.0, 1.0, 1.0])
        fixed.SetOrigin([0.0, 0.0, 0.0])

        moving = sitk.GetImageFromArray(arr)
        moving.SetSpacing([1.0, 1.0, 1.0])
        moving.SetOrigin([0.0, 0.0, 0.0])

        rmethod = build_registration_method()
        result = register_image(fixed, moving, rmethod)

        assert result.GetSize() == fixed.GetSize()
        assert result.GetPixelID() == sitk.sitkInt16
