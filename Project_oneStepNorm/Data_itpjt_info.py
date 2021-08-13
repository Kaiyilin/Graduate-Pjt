
import functions.Basic_tool as bst
import numpy as np 
import nibabel as nib 
import logging, os
from nilearn.image import resample_img
import matplotlib.pyplot as plt


logging.basicConfig(filename = 'Data_info_stand.log', level = logging.WARNING, format = '%(message)s')

data_dict = {"normalised": "./normalized/",
             "raw": "./raw/"}


raw_file_list = [f for f in os.listdir(data_dict["raw"]) if not f.startswith('.')]
raw_file_list.sort()
norm_file_list = [f for f in os.listdir(data_dict["normalised"])]
norm_file_list.sort()

logging.warning(f'Raw')
j = 0
for i in range(len(raw_file_list)):
    img = nib.load(os.path.join(data_dict["raw"], raw_file_list[i]))
    imag_array = img.get_fdata()
    logging.warning(f'Filename: {raw_file_list[i]}, shape: {imag_array.shape}, range[{imag_array.min():.2f}, {imag_array.max():.2f}]') 
    if imag_array.min() < 0:
        j += 1

logging.warning(f'\nTotal subjects that contains negative minimum value: {j}')

logging.warning(f'\nRaw_standardised')
for i in range(len(raw_file_list)):
    img = nib.load(os.path.join(data_dict["raw"], raw_file_list[i]))
    imag_array = img.get_fdata()
    #mean = imag_array.mean()
    imag_array = imag_array - imag_array.min()
    imag_array = bst.standardised(imag_array)
    logging.warning(f'Filename: {raw_file_list[i]}, shape: {imag_array.shape}, range after rescaling[{imag_array.min():.2f}, {imag_array.max():.2f}]') 

logging.warning(f'\nRaw_standardised_resample')
for i in range(len(raw_file_list)):
    img = nib.load(os.path.join(data_dict["raw"], raw_file_list[i]))
    img = resample_img(img, target_affine=np.eye(3)*2., interpolation='nearest')
    affine = img.affine
    imag_array = img.get_fdata()
    #mean = imag_array.mean()
    imag_array = imag_array - imag_array.min()
    imag_array = bst.standardised(imag_array)
    logging.warning(f'Filename: {raw_file_list[i]}, reshape: {imag_array.shape}, range after resample and rescaling[{imag_array.min():.2f}, {imag_array.max():.2f}]') 


logging.warning(f'\nNomalised')
for i in range(len(norm_file_list)):
    img = nib.load(os.path.join(data_dict["normalised"], norm_file_list[i]))
    imag_array = img.get_fdata()
    imag_array = imag_array - imag_array.min()
    logging.warning(f'Filename: {raw_file_list[i]}, shape: {imag_array.shape}, range[{imag_array.min():.2f}, {imag_array.max():.2f}]') 




