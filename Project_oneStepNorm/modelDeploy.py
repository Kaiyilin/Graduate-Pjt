# Deploy Deep Learning Model
from pandas.core.series import Series
import tensorflow as tf 
import numpy as np
import pandas as pd
import os, sys
import nibabel as nib 
from nilearn.image import resample_img

def standardised(image): 
    img = (image - image.mean())/image.std() 
    return img 

def padding_zeros(array, pad_size, channel_last = True):
    # define padding size
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



def CleanArray(array):
    # clean x 
    NewArray = np.delete(array, np.s_[0:19:1], axis = 1)
    NewArray = np.delete(NewArray, np.s_[-1:-19:-1], axis = 1)

    # clean y
    NewArray = np.delete(NewArray, np.s_[0:10:1], axis = 2)
    NewArray = np.delete(NewArray, np.s_[-1:-10:-1], axis = 2)
    # clean z
    NewArray = np.delete(NewArray, np.s_[0:19:1], axis = 3)
    NewArray = np.delete(NewArray, np.s_[-1:-19:-1], axis = 3)
    return NewArray

Data_Path = input("Given a file directory: ")
src_file_list = [f for f in os.listdir(Data_Path) if f.endswith(".nii")]
src_file_list.sort()
src_img_list = []
desired_file_list = [] 
undesired_file_list = []


trg_img = nib.load("./hdraffine.nii")
OutputPath = Data_Path + "Outputs"

try: 
    os.mkdir(OutputPath)
except:
    print("\n The output directory exist")



for i in range(len(src_file_list)):
    img = nib.load(os.path.join(Data_Path, src_file_list[i]))
    img = resample_img(img, target_affine=np.eye(3)*2., interpolation='nearest')
    imag_array = img.get_fdata()
    if imag_array.ndim == 4 and imag_array.shape[0] <= 128 and imag_array.shape[1] <= 128 and imag_array.shape[2] <= 128: 
        imag_array = standardised(imag_array)
        imag_array = imag_array - imag_array.min()
        padded_imag_array = padding_zeros(imag_array, pad_size=128)
        padded_imag_array = padded_imag_array[None,...]
        src_img_list.append(padded_imag_array)
        desired_file_list.append(src_file_list[i])
    elif imag_array.ndim ==3 and imag_array.shape[0] <= 128 and imag_array.shape[1] <= 128 and imag_array.shape[2] <= 128:
        imag_array = imag_array[...,None]
        imag_array = standardised(imag_array)
        imag_array = imag_array - imag_array.min()
        padded_imag_array = padding_zeros(imag_array, pad_size=128)
        padded_imag_array = padded_imag_array[None,...]
        src_img_list.append(padded_imag_array)
        desired_file_list.append(src_file_list[i])
    else:
        undesired_file_list.append(src_file_list[i])

series = pd.Series(undesired_file_list)
series.to_csv(os.path.join(OutputPath, "Report_Unfitted_Files.csv"))
print(f""" \033[33m 
The record of files which currently not support with this model is saved in \033[36m{OutputPath}\033[00m.
\033[33mI apologised for the inconvinience, this issue might be fixed in next version.\033[00m \n """)

src_imgs = np.concatenate(src_img_list, axis = 0)

Generator = tf.keras.models.load_model('./Models')
prdct = Generator.predict(src_imgs, batch_size=1, verbose = 1)

try:
    prdct.numpy()
    cleanedprdct = CleanArray(prdct)

    for i in range(len(cleanedprdct)):
        clipped_img = nib.Nifti1Image(cleanedprdct[i], trg_img.affine, trg_img.header)
        nib.save(clipped_img, os.path.join(OutputPath, f"Normed_{desired_file_list[i]}"))
except:
    cleanedprdct = CleanArray(prdct)

    for i in range(len(cleanedprdct)):
        clipped_img = nib.Nifti1Image(cleanedprdct[i], trg_img.affine, trg_img.header)
        nib.save(clipped_img, os.path.join(OutputPath, f"Normed_{desired_file_list[i]}"))
