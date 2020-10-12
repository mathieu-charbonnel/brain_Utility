import os
import argparse
import SimpleITK as sitk
from utils.parser import _parse, T1_exists, T2F_exists
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
  # Checks if both T1 and T2 Flair Sequences exist in the folder
  def allSequencesExist(self, image_dirs):
    t1_found = False
    t2_found = False

    for image_dir in image_dirs:
      t1_found = t1_found or T1_exists(image_dir)
      t2_found = t2_found or T2F_exists(image_dir)

    return t1_found and t2_found

  # Only for FLAIR Images
  # Registers all T2 Flair images to a Flair Atlas, and then creates
  # a mask given a mask file.
  def registerToAtlasFlair(self):

    # Atlas, Mask and Script files necessary for Registration
    script = os.getcwd() + '/utils/pipeline_mask_FLAIR_patients_v2.sh'

    filetree = _parse(self.data_path, MRI_type='T2F')

    for patient, dates in filetree.items():
      print("Patient:", patient)

      # Iterate through all dates and register images to the atlas image
      for date, files in dates.items():
        print(date)
        for file_dir in files:
          # Strip file from extension
          if file_dir.endswith(".nii.gz"):
            file_dir = os.path.splitext(os.path.splitext(file_dir)[0])[0]
            ext = "nii.gz"
          elif file_dir.endswith(".nii"):
            file_dir = os.path.splitext(file_dir)[0]
            ext = "nii"

          # Run Dr. Angelini's Script
          # Calls the registration script with the arguments
          filename = file_dir.split('/')[-1]
          file_dir = file_dir.rsplit('/', 1)[0]
          print("{}/{}".format(file_dir, filename))
          val = subprocess.check_call(script + ' %s %s %s %s' % (self.data_path, file_dir, filename, ext), shell=True)


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

    filetree = _parse(self.data_path)
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
        if self.allSequencesExist(files):
          first_date, image_dirs = date, files
          break
      # Remove this date from the dates so that we don't register to itself

      del dates[first_date]
      print("Using as First Date:", first_date)

      # Create fixed paths and images
      fixed_path_t1 = [x for x in image_dirs if T1_exists(x)][0]
      fixed_path_t2 = [x for x in image_dirs if T2F_exists(x)][0]
      fixed_image_t1 = sitk.ReadImage(fixed_path_t1, sitk.sitkFloat32)
      fixed_image_t2 = sitk.ReadImage(fixed_path_t2, sitk.sitkFloat32)
      print("Fixed Image T1:", fixed_path_t1)
      print("Fixed Image T2:", fixed_path_t2)

      # Iterate through all other dates and register images to the fixed image
      for date, files in dates.items():
        for file in files:
          moving_path = file
          moving_image = sitk.ReadImage(moving_path, sitk.sitkFloat32)
          print("Moving Image:", moving_path)

          # Check Sequence Type and set fixed image accordingly
          if T1_exists(moving_path):
            fixed_image = fixed_image_t1
          else:
            fixed_image = fixed_image_t2

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
            print('-> resampled')

            sitk.WriteImage(moving_image, moving_path)
            print('-> exported')

          except Exception as e:
            print("Error: Could not register image:", moving_path)
            print(e)

def main(args):
  copy_tree(args.rootdir, args.destinationdir)
  c = Coregister(args.destinationdir)

  if not args.f:
    # Co-register between images
    print("Co-registering images for each patient")
    c.coregister()
  else:
    # Co-register with atlas
    print("Registering with Flair Atlas:", args.f)
    c.registerToAtlasFlair()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('rootdir', help='Directory of Data')
  parser.add_argument('-f', action='store_true', help='Register to a Flair Atlas and Mask (Flair images only, T1 remains unaffected)')
  parser.add_argument('destinationdir', help='Directory where we intend to store the registered data')

  main(parser.parse_args())
