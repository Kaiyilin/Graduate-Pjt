import os
import sys
import numpy as np
import nibabel as nib
import tensorflow as tf
from nilearn.image import resample_img
# Data Dir 
data_dict = {
    "normalised": "./morm",
    "raw": "./raw"
    }


# Data Preprocessing
def data_preprocessing(image):
    image = (image - image.min())/(image.max() - image.min()) 
    return image

def standardised(image): 
    img = (image - image.mean())/image.std() 
    return img 



# Data_preparation using tf data
raw_file_list = [f for f in os.listdir(data_dict["raw"]) if not f.startswith('.')]
raw_file_list.sort()

raw_img_list = []
desired_file_list = []

# Only data with image shape < (128, 128, 128)
for i in range(len(raw_file_list)):
    img = nib.load(os.path.join(data_dict["raw"], raw_file_list[i]))
    img = resample_img(img, target_affine=np.eye(3)*2., interpolation='nearest')
    imag_array = img.get_fdata()
    if imag_array.shape[0] <= 128: 
        imag_array = standardised(imag_array)
        imag_array = imag_array - imag_array.min()
        padded_imag_array = padding_zeros(imag_array, pad_size=128)
        padded_imag_array = padded_imag_array[None,...]
        raw_img_list.append(padded_imag_array)
        desired_file_list.append(raw_file_list[i])
    else:
        pass

# Labels of selected raw data
normed_file_list = [] 
for i in range(len(desired_file_list)):
    img = nib.load(os.path.join(data_dict["normalised"], desired_file_list[i]))
    imag_array = img.get_fdata()
    imag_array = imag_array - imag_array.min()
    padded_imag_array = padding_zeros(imag_array, pad_size=128)
    padded_imag_array = padded_imag_array[None,...]
    normed_file_list.append(padded_imag_array)

raw_img = np.concatenate(raw_img_list, axis = 0)
raw_img = raw_img.astype('float32')
nor_img = np.concatenate(normed_file_list, axis = 0)
nor_img = nor_img.astype('float32')


# Make sure the data fits criteria
if raw_img.shape == nor_img.shape and nor_img.dtype == raw_img.dtype:
    ds = tf.data.Dataset.from_tensor_slices((raw_img, nor_img))
else:
    print("Shape of source img: ", raw_img.shape)
    print("Shape of target img: ", nor_img.shape)
    print("Type of source img: ", raw_img.dtype)
    print("Type of target img: ", nor_img.dtype)
    sys.exit("\033[93m  The size or type of source and target unmatched, check the size again \033[00m")

# Flush out unnecessary memory usage
del imag_array, padded_imag_array, img, raw_file_list, normed_file_list, desired_file_list, raw_img, nor_img