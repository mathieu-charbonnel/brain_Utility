# -*- coding: utf-8 -*-
import SimpleITK as sitk
import os
import numpy as np
from datetime import datetime
import argparse


class PreProcess(object):
    data_path = None
    def __init__(self, data_path):
        if data_path != None:
            self.data_path = data_path

    def rescale(self):
        for root, dirs, files in os.walk(self.data_path):
            for name in files:
                file_path=os.path.join(root, name)
                print("file:",file_path)
                img = sitk.ReadImage(file_path)
                img_arr = sitk.GetArrayFromImage(img)

                #max=np.amax(img_arr)
                #min=np.amin(img_arr)

                #max=2058
                #min=0

                max=1989
                min=-1000

                diff=(max-min)/2.0
                middle=(max+min)/2.0
                img_arr=(img_arr-middle)/diff

                print("rescaled")
                new_nii = sitk.GetImageFromArray(img_arr)
                sitk.WriteImage(new_nii, file_path)



def main(args):

	pp = PreProcess(data_path=args.rootdir)

	pp.rescale()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('rootdir', help='Directory of Data')

	main(parser.parse_args())
