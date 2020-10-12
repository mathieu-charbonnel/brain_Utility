import os
import argparse
import SimpleITK as sitk
from sklearn.metrics import mean_absolute_error, mean_squared_error
from skimage.metrics import structural_similarity as ssim
from utils.parser import get_file_paths, separate_rA_rB_fB
import numpy as np
import time
import progressbar


class Evaluation(object):
	# Default Settings
	# Root directory where data is stored
	data_path = '/data/MSc_students_accounts/mathieu/MscProject/Ea-GANs/results/example_gEaGAN20/test_latest/images'
	file_paths = None
	MAX_pixel = 100 # For 8-bit images

	def __init__(self, data_path):

		self.image_height = 128
		self.image_width = 128
		self.image_depth = 128
		if data_path != None:
			self.data_path = data_path

	# Loads MRI Slice data after being exported from the test function
	# of the network
	def loadData(self):
		print("Loading from:{}\n".format(self.data_path))
		self.file_paths = mc_get_file_paths(self.data_path)

		# Check if data was loaded
		if len(self.file_paths) <= 0:
			print("No image files found")
			return


		self.rA,self.rB,self.fB = separate_rA_rB_fB(self.file_paths)


		#sitk.ReadImage(img_coregistered)
		#savImg = sitk.GetImageFromArray(image_tensor

		sample_no = len(self.rA)

		# Create numpy array to hold data
		dataset_fake = np.ndarray(shape=(sample_no,  self.image_depth * self.image_height * self.image_width),
                     dtype=np.float32)
		dataset_real = np.ndarray(shape=(sample_no, self.image_depth * self.image_height * self.image_width),
                     dtype=np.float32)
		dataset_input = np.ndarray(shape=(sample_no, self.image_depth * self.image_height * self.image_width),
                     dtype=np.float32)


		for i in range(len(self.rA)):
			real_A=self.rA[i]
			real_B=self.rB[i]
			fake_B=self.fB[i]
			# Read images as numpy array
			fake_B_im = sitk.ReadImage(fake_B)
			fake_B_im=  sitk.GetArrayFromImage(fake_B_im)
			real_B_im = sitk.ReadImage(real_B)
			real_B_im=  sitk.GetArrayFromImage(real_B_im)
			real_A_im = sitk.ReadImage(real_A)
			real_A_im=  sitk.GetArrayFromImage(real_A_im)

			# Put images into dataset
			dataset_fake[i] = fake_B_im.reshape(-1)
			dataset_real[i] = real_B_im.reshape(-1)
			dataset_input[i] = real_A_im.reshape(-1)
		#rescale the images as they are usually between -1 and 1
		dataset_fake = (dataset_fake + 1) / 2.00 * 255.0
		dataset_real = (dataset_real + 1) / 2.00 * 255.0
		dataset_input = (dataset_input + 1) / 2.00 * 255.0
		return dataset_fake, dataset_real, dataset_input

	def mae_underthreshold(self,img1, img2):
		threshold = 100
		sigma = lambda x: np.array((x < threshold), dtype=np.float32)
		f = lambda x, y: ((sigma(x) * sigma(y))*np.abs(x-y))/np.sum(sigma(x)*sigma(y))
		return np.sum(f(img1, img2))

	def mae_overthreshold(self,img1, img2):
		threshold = 100
		sigma = lambda x: np.array((x > threshold), dtype=np.float32)
		f = lambda x, y: (np.logical_or(sigma(x),sigma(y))*np.abs(x-y))/np.sum(np.logical_or(sigma(x),sigma(y)))
		return np.sum(f(img1, img2))


#
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
	def Reconstruction_score(self, data_fake, data_input):

		num_samples = data_fake.shape[0]
		score=0

		# Calculate a score for every fake sample, by calculating
		# the distance between every other input and real sample
		# and checking which ones are the smallest ones
		for i in progressbar.progressbar(range(0, num_samples)):

			distance_matrix = np.zeros((num_samples))

			distance_corresponding=self.mae_underthreshold(data_input[i], data_fake[i])
			for j in range(0, num_samples):
				distance_matrix[j] = self.mae_underthreshold(data_input[j], data_fake[i])

			# Get k smallest elements indices
			k = 2
			idx = np.argpartition(distance_matrix, k)[:k]
			idx = idx[np.argsort(distance_matrix[idx])]


			if np.abs(distance_matrix[idx[0]] - distance_corresponding) < 0.01 or np.abs(distance_matrix[idx[1]] - distance_corresponding) < 0.1 :
				score += 1


		rs = score/num_samples
		return rs



	def Evolution_score(self, data_fake, data_real):

		num_samples = data_fake.shape[0]
		score=0

		# Calculate a score for every fake sample, by calculating
		# the distance between every other input and real sample
		# and checking which ones are the smallest ones
		for i in progressbar.progressbar(range(0, num_samples)):

			distance_matrix = np.zeros((num_samples))

			distance_corresponding=self.mae_overthreshold(data_real[i], data_fake[i])
			for j in range(0, num_samples):
				distance_matrix[j] = self.mae_overthreshold(data_real[j], data_fake[i])

			# Get 2 smallest elements indices
			k = 2
			idx = np.argpartition(distance_matrix, k)[:k]
			idx = idx[np.argsort(distance_matrix[idx])]


			if np.abs(distance_matrix[idx[0]] - distance_corresponding)<0.1 or np.abs(distance_matrix[idx[1]] - distance_corresponding)  < 0.1 :
				score += 1

		es = score/num_samples
		return es



#-------------------------------------------------------------------------------------------------------------------------------------------------------------------

	def MAE(self, data_fake, data_real):
		return mean_absolute_error(y_true=data_real, y_pred=data_fake)

	def MSE(self, data_fake, data_real):
		return mean_squared_error(y_true=data_real, y_pred=data_fake)

	def PSNR(self, data_fake, data_real):
		mse = self.MSE(data_fake, data_real)
		return 20 * np.log10(self.MAX_pixel) - 10 * np.log10(mse)

	def SSIM(self, data_fake, data_real):
		cum_ssim = 0
		for i in progressbar.progressbar(range(0, data_fake.shape[0])):
			cum_ssim += ssim(data_real[i], data_fake[i])

		rssim = cum_ssim / data_fake.shape[0]
		return rssim



def main(args):

	print("Evaluating metrics:")

	ev = Evaluation(args.rootdir)
	data_fake, data_real, data_input = ev.loadData()

	# Calculate and Print Metrics
	mae = ev.MAE(data_fake, data_real)
	mse = ev.MSE(data_fake, data_real)
	#psnr = ev.PSNR(data_fake, data_real)
	ssim = ev.SSIM(data_fake, data_real)
	rs= ev.Reconstruction_score(data_fake, data_input)
	es= ev.Evolution_score(data_fake, data_real)

	print("MAE:", mae)
	print("MSE:", mse)
	#print("PSNR:", psnr)
	print("SSIM:", ssim)
	print("Reconstruction Score",rs)
	print("Evolution Score",es)



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('rootdir', help='Directory of Data')

	main(parser.parse_args())
