import os
import sys
import numpy as np
import nibabel as nib
import tensorflow as tf
from configs import prjt_configs
from nilearn.image import resample_img


# Data Preprocessing
def data_preprocessing(image):
    image = (image - image.min())/(image.max() - image.min()) 
    return image

def standardised(image): 
    img = (image - image.mean())/image.std() 
    return img 

def pad_to_sqr(array, pad_size, channel_last=True):
    
    elements = array.shape    
    for element in elements:
        if element > pad_size:
            sys.exit('\nThe expanded dimension shall be greater than your current dimension')
    pad_list = list() 
    if channel_last == True:
        for i in range(array.ndim - 1):
            x = pad_size - array.shape[i]
            if x%2 ==1:
                y_1 = (x/2 +0.5)
                y_2 = (x/2 -0.5)
                z = (int(y_1),int(y_2))
                pad_list.append(z)

            else:
                y = int(x/2)
                z=(y,y)
                pad_list.append(z)
    pad_list.append((0,0))
    pad_array = np.pad(array, pad_list, 'constant')
    pad_list = list() 
    return pad_array

# Data_preparation using tf data
raw_file_list = [f for f in os.listdir(prjt_configs["data"]["raw"]) if not f.startswith('.')]
raw_file_list.sort()

raw_img_list = []
desired_file_list = []

# Only data with image shape < (128, 128, 128)
for i in range(len(raw_file_list)):
    img = nib.load(os.path.join(prjt_configs["data"]["raw"], raw_file_list[i]))
    img = resample_img(img, target_affine=np.eye(3)*2., interpolation='nearest')
    imag_array = img.get_fdata()
    if imag_array.shape[0] <= 128: 
        imag_array = standardised(imag_array)
        imag_array = imag_array - imag_array.min()
        padded_imag_array = pad_to_sqr(imag_array, pad_size=128)
        padded_imag_array = padded_imag_array[None,...]
        raw_img_list.append(padded_imag_array)
        desired_file_list.append(raw_file_list[i])
    else:
        pass

# Labels of selected raw data
normed_file_list = [] 
for i in range(len(desired_file_list)):
    img = nib.load(os.path.join(prjt_configs["data"]["normalised"], desired_file_list[i]))
    imag_array = img.get_fdata()
    imag_array = imag_array - imag_array.min()
    padded_imag_array = pad_to_sqr(imag_array, pad_size=128)
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