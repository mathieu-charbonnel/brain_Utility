import os
import argparse
import SimpleITK as sitk
from utils.parser import _parse, T1_exists, T2F_exists
from datetime import datetime
import subprocess
from distutils.dir_util import copy_tree
from utils.parser import _parse, _parse_BRATS, mc_get_file_paths
import numpy as np
import nibabel
from nibabel import processing
voxel_size_val = 1


def get_get(folder, masks=False):
    image_file_paths = []
    for root, dirs, filenames in os.walk(folder):
        filenames = sorted(filenames)
        for filename in filenames:
            input_path = os.path.abspath(root)
            file_path = os.path.join(input_path, filename)
            if filename.endswith('.nii') or filename.endswith('.nii.gz'):
                if not masks and '_mask' in filename:
                  continue
                elif masks and '_mask' not in filename:
                  continue
                image_file_paths.append(file_path)
    return image_file_paths

#if you need a copy of the old (not upsampled) images
#copy_tree('/data/MSc_students_accounts/mathieu/MscProject/brain_data/mc_coregistered_data', '/data/MSc_students_accounts/mathieu/MscProject/brain_data/mc_upsampled_data')


# Resample without any smoothing.
#Choose the directory you need to perform the resampling on
for file_path in get_get("/data/MSc_students_accounts/mathieu/MscProject/brain_data/64_Data/"):
    img = sitk.ReadImage(file_path)
    #from https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/blob/master/Python/70_Data_Augmentation.ipynb
    #Normal size is [128,128,128] but use [64,64,64] data for preparing 64_dataset.
    new_size = [64, 64, 64]
    reference_image = sitk.Image(new_size, img.GetPixelIDValue())
    reference_image.SetOrigin(img.GetOrigin())
    reference_image.SetDirection(img.GetDirection())
    reference_image.SetSpacing([sz*spc/nsz for nsz,sz,spc in zip(new_size, img.GetSize(), img.GetSpacing())])

    imgout = sitk.Resample(img, reference_image)

    sitk.WriteImage(imgout, file_path)
