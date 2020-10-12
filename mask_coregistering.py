import os
import argparse
import SimpleITK as sitk
from utils.parser import _parse_tumor_masking, T1_exists, T2F_exists, segmentation_exists
from datetime import datetime
import subprocess
from distutils.dir_util import copy_tree

class Coregister(object):
  # Root directory where data is stored
  data_path = None # Default
  destination_path= None #Default

  def __init__(self, data_path):
    if data_path != None:
      self.data_path = data_path

  def coregister(self):
    # Code adapted from: https://github.com/bodokaiser/mrtoct-scripts
    rmethod = sitk.ImageRegistrationMethod()
    rmethod.SetMetricAsMattesMutualInformation(numberOfHistogramBins=100)
    rmethod.SetMetricSamplingStrategy(rmethod.RANDOM)
    rmethod.SetMetricSamplingPercentage(.01)
    rmethod.SetInterpolator(sitk.sitkLinear)
    rmethod.SetOptimizerAsGradientDescent(learningRate=1.0,
                                          numberOfIterations=200,
                                          convergenceMinimumValue=1e-6,
                                          convergenceWindowSize=10)
    rmethod.SetOptimizerScalesFromPhysicalShift()
    rmethod.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    rmethod.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    rmethod.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    filetree = _parse_tumor_masking(self.data_path)
    print(filetree)

    for patient, dates in filetree.items():
      print("Patient:", patient)
      # We want to register all images of the other days with one of the baseline day.
      # We register T1 to T1 and T2 to T2

      # Sort Dates
      dates_list = []
      for date, _ in dates.items():
        dates_list.append(date)
      date_format = '%d.%m.%y'
      dates_list = sorted(dates_list, key=lambda x: datetime.strptime(x, date_format))
      print("Dates List:")
      print(dates_list)

      # Get the first date folder which includes both T1 and T2 sequences
      # that we can use as our fixed images for registration.
      first_date = None
      image_dirs = None
      for date in dates_list:
        files = dates[date]
        first_date, image_dirs = date, files
        break
      # Remove this date from the dates so that we don't register to itself

      del dates[first_date]
      print("Using as First Date:", first_date)

      # Create fixed paths and images
      fixed_path = [x for x in image_dirs if T1_exists(x)][0]
      fixed_seg_path = [x for x in image_dirs if segmentation_exists(x)][0]
      fixed_image = sitk.ReadImage(fixed_path, sitk.sitkFloat32)
      fixed_seg = sitk.ReadImage(fixed_seg_path, sitk.sitkFloat32)
      print("Fixed Image T1:", fixed_path)
      print("Fixed segmentation :", fixed_seg_path)

      # Iterate through all other dates and register images to the fixed image
      for date, files in dates.items():
        for file in files:
          if T1_exists(file):
              moving_path = file
              moving_image = sitk.ReadImage(moving_path, sitk.sitkFloat32)
          if segmentation_exists(file):
              moving_segmentation_path=file
              moving_segmentation=sitk.ReadImage(moving_segmentation_path, sitk.sitkFloat32)

        print("Moving Image:", moving_path)
        print("Moving Segmentation", moving_segmentation_path)



        initial_transform = sitk.CenteredTransformInitializer(
        fixed_image, moving_image, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)
        rmethod.SetInitialTransform(initial_transform, inPlace=False)

        try:
            final_transform = rmethod.Execute(fixed_image, moving_image)
            print('-> coregistered')

            moving_image = sitk.Resample(
            moving_image, fixed_image, final_transform, sitk.sitkLinear, .0,
            moving_image.GetPixelID())


            moving_image = sitk.Cast(moving_image, sitk.sitkInt16)
            print('-> t1 resampled')

            sitk.WriteImage(moving_image, moving_path)
            print('-> t1 exported')

            moving_segmentation = sitk.Resample(
            moving_segmentation, fixed_seg, final_transform, sitk.sitkLinear, .0,
            moving_segmentation.GetPixelID())


            moving_segmentation = sitk.Cast(moving_segmentation, sitk.sitkInt16)
            print('-> segmentation resampled')

            sitk.WriteImage(moving_segmentation, moving_segmentation_path)
            print('-> segmentation exported')

        except Exception as e:
            print("Error: Could not register image:", moving_path)
            print(e)




def main(args):
  copy_tree(args.rootdir, args.destinationdir)
  c = Coregister(args.destinationdir)
  print("Co-registering images for each patient")
  c.coregister()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('rootdir', help='Directory of Data')
  parser.add_argument('destinationdir', help='Directory where we intend to store the registered data')

  main(parser.parse_args())
