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
from skimage.transform import match_histograms


class PreProcess(object):
	# Root directory where data is stored
	data_path = None # Default
	# Change to your spreadsheet location

	def __init__(self, data_path):
		if data_path != None:
			self.data_path = data_path

	# Converts (by making a copy) every *.mhd file in a directory to a *.nii file
	def convertMhdToNii(self):
		count = 0

		for root, dirs, files in os.walk(self.data_path):
			for file in files:
				if file.endswith(".mha"):
					filename = os.path.splitext(file)[0]
					file_dir = os.path.join(root, file)
					nii_path = root + '/' + filename + '.nii'

					img = sitk.ReadImage(file_dir)
					sitk.WriteImage(img, nii_path)
					count += 1
					print("Converted:", file)

		print("Converted {} files".format(count))

	# Extracts a bounding box from a given MRI image and saves the
	# extracted volume in the same directory as the file
	def extractBB(self, filepath, coordinates=(0,0,0), volume=(50, 40, 10)):
		img = sitk.ReadImage(filepath)
		img = img[::-1, ::-1, :] # Flip Image to have orientation like Mango App

		# Extract tumour coordinates from input
		(x, y, z) = coordinates
		size_x, size_y, size_z = img.GetSize()
		z = size_z - z # Fix for correcting z coordinate in relation to Mango App
		hv_x, hv_y, hv_z = [int(v/2) for v in volume]

		# display_image(img, x=x, y=y, z=z, window=1000, level=400)
		img_array = sitk.GetArrayFromImage(img)

		# Calculate boundaries of bounding box
		low_z = max(0, min(z - hv_z, size_z))
		high_z = max(0, min(z + hv_z, size_z))
		low_y = max(0, min(y - hv_y, size_y))
		high_y = max(0, min(y + hv_y, size_y))
		low_x = max(0, min(x - hv_x, size_x))
		high_x = max(0, min(x + hv_x, size_x))

		print("Z: [{} : {}]".format(low_z, high_z))
		print("Y: [{} : {}]".format(low_y, high_y))
		print("X: [{} : {}]".format(low_x, high_x))

		bb = np.zeros(img_array.shape)
		bb[low_z : high_z, low_y : high_y, low_x : high_x] = 400
		display_image(img, bb, x=x, y=y, z=z, window=1000, level=400)

		# Extract and save volume
		seg = img[low_x : high_x, low_y : high_y, low_z : high_z]
		display_image(seg, window=1000, level=400)
		new_filename = '.'.join(filepath.split('/')[-1].split('.')[:-1]) + '_tumour_extracted.nii'

		save = input("Do you wish to save and extract this volume? y/n: ")
		if save == 'y':
			file_folder = '/'.join(filepath.split('/')[:-1])
			sitk.WriteImage(seg, os.path.join(file_folder, new_filename))
			print("saving in:", os.path.join(file_folder, new_filename))
			print("Volume saved")
		else:
			print("Volume not saved")

	# Performs histogram matching of all MRIs in dataset of sequence
	# type MRI_type, using given template flair
	def performHM(self, MRI_type='T2F'):
		filetree = _parse(self.data_path, MRI_type=MRI_type)

		# Flair Template
		flair_template_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'files/Mean_Brain_Total_23_1mm_noVentr_NL_reg_edited.nii.gz')
		ref = sitk.ReadImage(flair_template_path)
		# Convert to array and flatten to 2D for histogram matching
		ref_arr = sitk.GetArrayFromImage(ref)
		ref_arr = ref_arr.reshape(-1, ref_arr.shape[2])

		# Iterate through file tree
		for patient, dates in filetree.items():
			print("Patient:", patient)

			# Iterate through all dates and perform histogram matching to template
			for date, files in dates.items():
				print(date)
				for file_dir in files:
					print("file:",file_dir)
					file_mask_dir = file_dir.rsplit('/', 1)[0] + '/alpha_mask_brain_roi.nii.gz'

					# Read Image files and convert to arrays
					img = sitk.ReadImage(file_dir)
					mask = sitk.ReadImage(file_mask_dir)
					img_arr = sitk.GetArrayFromImage(img)
					mask_arr = sitk.GetArrayFromImage(mask)

					# Flatten to 2D for Histogram Matching
					img_arr_dim0 = img_arr.shape[0]
					img_arr = img_arr.reshape(-1, img_arr.shape[2])

					# Histogram Matching
					matched = match_histograms(img_arr, ref_arr, multichannel=False)

					# Return image back to original shape
					matched = np.ceil(matched.reshape(img_arr_dim0, -1, img_arr.shape[1]))

					# Mask Background Values
					matched[mask_arr <= 0] = 0

					# Copy Nifti file settings and save file
					new_nii = sitk.GetImageFromArray(matched)
					new_nii.CopyInformation(img)
					new_nii = sitk.Cast(new_nii, sitk.sitkFloat32)
					sitk.WriteImage(new_nii, file_dir)




	# Creates a folder with image slices for every *.nii file in the directory
	# Then combines the slices to be side by side, and creates a train-val-test split
	# To be used with Pix2Pix and our private dataset
	def createPairedImages(self, MRI_type, days=(None, None), store_mask=False, split_perc=0.7):

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

		if store_mask:
			masks_dir = new_dir + 'masks/'
			if not os.path.exists(masks_dir):
				os.makedirs(masks_dir)

		# Iterate through dictionary and create sliced pairs
		for patient, dates in filetree.items():
			print(patient)

			# Sort dates and create permutations of 2-tuples
			date_format = '%d.%m.%y'
			dates_list = sorted(dates.keys(), key=lambda x: datetime.strptime(x, date_format))
			combs = list(it.combinations(dates_list, 2))

			for (date1, date2) in combs:

				date1_date = datetime.strptime(date1, date_format)
				date2_date = datetime.strptime(date2, date_format)
				diff = (date2_date - date1_date).days

				# Check if we are going from early to later day
				assert(diff > 0)

				# Check if Date Range is within limits:
				(low_day, high_day) = days
				if low_day != None and high_day != None:
					# If outside range, skip
					if diff < low_day or diff > high_day:
						continue

				img1_path = dates[date1][0]
				img2_path = dates[date2][0]
				print("Date1: {}, Date2: {}".format(date1, date2))
				print("img1: {}, img2: {}".format(img1_path, img2_path))
				img1 = sitk.ReadImage(img1_path)
				img2 = sitk.ReadImage(img2_path)

				if store_mask:
					mask1_path = img1_path.rsplit('/', 1)[0] + '/alpha_mask_brain_roi.nii.gz'
					mask2_path = img2_path.rsplit('/', 1)[0] + '/alpha_mask_brain_roi.nii.gz'
					mask1 = sitk.ReadImage(mask1_path)
					mask2 = sitk.ReadImage(mask2_path)


				path_A = A_dir + str(counter) + '_' + str(diff // 7) + 'w.nii'
				path_B = B_dir + str(counter) + '_' + str(diff // 7) + 'w.nii'
				try:
					sitk.WriteImage(img1, path_A)
					sitk.WriteImage(img2, path_B)
				except IndexError as e:
				# In this case the images are not registered so they have
				# different z-depth. Skip these as we can't create pairs
					continue

				# Mask Files
				if store_mask:
					path_A_mask = A_dir + str(counter) + '_mask_' + str(diff // 7) + 'w.nii'
					path_B_mask = B_dir + str(counter) + '_mask_' + str(diff // 7) + 'w.nii'

					try:
						sitk.WriteImage(mask1, path_A_mask)
						sitk.WriteImage(mask2, path_B_mask)
					except IndexError as e:
					# In this case the images are not registered so they have
					# different z-depth. Skip these as we can't create pairs
						continue

				counter += 1


				#print("Saved Image Pair as {} slices:".format(high-low))

		# Create combined images
		a_paths = get_file_paths(A_dir)
		b_paths = get_file_paths(B_dir)
		self.align_images(a_paths, b_paths, AB_dir)


		# Create combined masks
		if store_mask:
			a_paths = get_file_paths(A_dir, masks=True)
			b_paths = get_file_paths(B_dir, masks=True)
			self.align_images(a_paths, b_paths, masks_dir)

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


	# Extended from Pix2Pix: Creates Aligned Images
	def align_images(self, a_file_paths, b_file_paths, target_path, time=True):
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
				print('img_a',img_a.GetSize())
				print('img_b',img_b.GetSize())'''

			time_period = a_file_paths[i].split('_')[-1].split('.')[0]
			sitk.WriteImage(aligned_image, os.path.join(target_path, '{:04d}_{}.nii'.format(i, time_period)))



	# Creates and returns a train-val-test split given a split_perc value
	def createTrainValTestSplit(self, data_path, split_perc):
		X_train, X_test = train_test_split(data_path, train_size=split_perc, random_state=42)
		X_val, X_test = train_test_split(X_test, train_size=0.5, random_state=42)
		return X_train, X_val, X_test

	# Deletes all MRI files of certain type in root directory
	# Used to clean up dataset of MRI sequence types we don't need
	def deleteByType(self, MRI_type):
		count = 0
		filetree = _parse(self.data_path, MRI_type=MRI_type)
		for patient, dates in filetree.items():
			for date, files in dates.items():
				for file in files:
					os.remove(file)
					count += 1
		print("Deleted {} files of type {}.".format(count, MRI_type))


def main(args):
	###############################################
	# Uncomment the function call you want to use #
	###############################################

	pp = PreProcess(data_path=args.rootdir)

	# pp.convertMhdToNii()
	# pp.extractBB(#FILEPATH#, coordinates=(#X#, #Y#, #Z#), volume=(#X#, #Y#, #Z#))
	# pp.performHM('T2F')
	pp.createPairedImages('T2F', store_mask=False)
	# pp.deleteByType('T1')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('rootdir', help='Directory of Data')

	main(parser.parse_args())
