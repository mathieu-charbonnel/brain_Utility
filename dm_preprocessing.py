# -*- coding: utf-8 -*-
import SimpleITK as sitk
import os, shutil
import numpy as np
from utils.image_viewer import display_image
import itertools as it
from utils.parser import _parse, _parse_BRATS, get_file_paths
from sklearn.model_selection import train_test_split
from datetime import datetime
import argparse
#import cv2


class PreProcess(object):
	# Root directory where data is stored
	data_path = None # Default

	def __init__(self, data_path):
		if data_path != None:
			self.data_path = data_path

	# Prepare the dataset for DM-gEAGAN model:
	# Find triplets of dates with scan from the same patient
	# compute the difference image1-imgag0
	# pair this deifference with image 2
	# add the ratio (t2-t1)/(t1-t0) to the name of the file
	# Creates a train-val-test split

	def createPairedImages(self, MRI_type, split_perc=0.7):

		# Type Checks
		if MRI_type != 'T1' and MRI_type != 'T2F':
			print("Incorrect MRI type given: Must be either 'T1' or 'T2F' (Flair)")
			return

		counter = 0
		filetree = _parse(self.data_path, MRI_type=MRI_type)

		# Make directories
		new_dir = self.data_path + '/pairing/'
		A_dir = new_dir + 'A/'
		B_dir = new_dir + 'B/'
		AB_dir = new_dir + 'AB/'



		if not os.path.exists(A_dir):
			os.makedirs(A_dir)
		if not os.path.exists(B_dir):
			os.makedirs(B_dir)
		if not os.path.exists(AB_dir):
			os.makedirs(AB_dir)

		for patient, dates in filetree.items():
			print(patient)

			# Sort dates and create permutations of 3-tuples
			date_format = '%d.%m.%y'
			dates_list = sorted(dates.keys(), key=lambda x: datetime.strptime(x, date_format))
			combs = list(it.combinations(dates_list, 3))

			for (date1, date2, date3) in combs:

				date1_date = (datetime.strptime(date1, date_format))
				date2_date = (datetime.strptime(date2, date_format))
				date3_date = (datetime.strptime(date3, date_format))

				diff1 = (date2_date - date1_date).days
				diff2 = (date3_date - date2_date).days

				diff1=float(diff1)
				diff2=float(diff2)
				#date ratio is rounded to be integrated to the file's name
				ratio=round(diff2/diff1,3)


				img1_path = dates[date1][0]
				img2_path = dates[date2][0]
				img3_path = dates[date3][0]
				print("Date1: {}, Date2: {}, Date3: {}".format(date1, date2, date3))
				print("img1: {}, img2: {}, img3: {}".format(img1_path, img2_path, img3_path))

				img1 = sitk.ReadImage(img1_path)
				img2 = sitk.ReadImage(img2_path)
				img3 = sitk.ReadImage(img3_path)

				arr1 = sitk.GetArrayFromImage(img1)
				arr2 = sitk.GetArrayFromImage(img2)

				#Compute the difference map between t2 and t1
				diff_arr = arr2-arr1
				diff_img = sitk.GetImageFromArray(diff_arr)


				path_A = A_dir + str(counter) + '_' + str(ratio) + 'r.nii'
				path_B = B_dir + str(counter) + '_' + str(ratio) + 'r.nii'
				try:
					sitk.WriteImage(diff_img, path_A)
					sitk.WriteImage(img3, path_B)
				except IndexError as e:
				# In this case the images are not registered so they have
				# different z-depth. Skip these as we can't create pairs
					continue

				counter += 1


				#print("Saved Image Pair as {} slices:".format(high-low))

		# Create combined images
		a_paths = mc_get_file_paths(A_dir)
		b_paths = mc_get_file_paths(B_dir)
		self.align_images(a_paths, b_paths, AB_dir, padding=False)



		try:
			# Create Train - Val - Split
			ab_path = mc_get_file_paths(AB_dir)
			train, val, test = self.createTrainValTestSplit(ab_path, split_perc)

			# Move data to corresponding directories
			train_dir = new_dir + 'train/'
			val_dir = new_dir + 'val/'
			test_dir = new_dir + 'test/'

			if not os.path.exists(train_dir):
				os.makedirs(train_dir)
			if not os.path.exists(val_dir):
				os.makedirs(val_dir)
			if not os.path.exists(test_dir):
				os.makedirs(test_dir)

			for file in train:
				shutil.move(file, train_dir)
			for file in val:
				shutil.move(file, val_dir)
			for file in test:
				shutil.move(file, test_dir)

			# Delete A, B, AB folders
			shutil.rmtree(A_dir)
			shutil.rmtree(B_dir)
			shutil.rmtree(AB_dir)
		except Exception as e:
			print("Error: Could not create train-val-test split")
			print(e)


	# Creates Aligned Images in 3D
	def align_images(self, a_file_paths, b_file_paths, target_path, padding=False):
		if not os.path.exists(target_path):
			os.makedirs(target_path)

		for i in range(len(a_file_paths)):
			img_a = sitk.ReadImage(a_file_paths[i])
			img_b = sitk.ReadImage(b_file_paths[i])

			img_a=sitk.Cast(img_a, sitk.sitkInt32)
			img_b=sitk.Cast(img_b, sitk.sitkInt32)


			aligned_image = sitk.Image(128, 128, 256, sitk.sitkInt32)

			try:
				aligned_image = sitk.Paste(aligned_image, img_a, (128,128,128) , destinationIndex=[0,0,0])
				aligned_image = sitk.Paste(aligned_image, img_b, (128,128,128) , destinationIndex=[0,0,128])
			except Exception as e:
				print('img_a:',img_a.GetSize())
				print('img_b:',img_b.GetSize())

			time_period = a_file_paths[i].split('_')[-1]
			sitk.WriteImage(aligned_image, os.path.join(target_path, '{:04d}_{}.nii'.format(i, time_period)))



	# Creates and returns a train-val-test split given a split_perc value
	def createTrainValTestSplit(self, data_path, split_perc):
		X_train, X_test = train_test_split(data_path, train_size=split_perc, random_state=42)
		X_val, X_test = train_test_split(X_test, train_size=0.5, random_state=42)
		return X_train, X_val, X_test



def main(args):

	pp = PreProcess(data_path=args.rootdir)
	pp.createPairedImages('T2F')



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('rootdir', help='Directory of Data')

	main(parser.parse_args())
