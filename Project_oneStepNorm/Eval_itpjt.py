from datetime import date
import functions.Basic_tool as bst
import tensorflow as tf
import sys, os
import nibabel as nib
import numpy as np
from nilearn.image import resample_img
import matplotlib.pyplot as plt
import pandas as pd
# Data Dir 
data_dict = {"normalised": "./normalized/",
             "raw": "./raw/"}



# Data_preparation using tf data
raw_file_list = [f for f in os.listdir(data_dict["raw"]) if not f.startswith('.')]
raw_file_list.sort()

raw_img_list = []
desired_file_list = []

for i in range(len(raw_file_list)):
    img = nib.load(os.path.join(data_dict["raw"], raw_file_list[i]))
    img = resample_img(img, target_affine=np.eye(3)*2., interpolation='nearest')
    imag_array = img.get_fdata()
    if imag_array.shape[0] <= 128: 
        #imag_array = imag_array - imag_array.min()
        imag_array = bst.standardised(imag_array)
        imag_array = imag_array - imag_array.min()
        padded_imag_array = bst.padding_zeros(imag_array, pad_size=128)
        padded_imag_array = padded_imag_array[None,...]
        raw_img_list.append(padded_imag_array)
        desired_file_list.append(raw_file_list[i])
    else:
        pass

normed_file_list = [] 
for i in range(len(desired_file_list)):
    img = nib.load(os.path.join(data_dict["normalised"], desired_file_list[i]))
    imag_array = img.get_fdata()
    imag_array = imag_array - imag_array.min()
    padded_imag_array = bst.padding_zeros(imag_array, pad_size=128)
    padded_imag_array = padded_imag_array[None,...]
    normed_file_list.append(padded_imag_array)
affine = img.affine

raw_img = np.concatenate(raw_img_list, axis = 0)
raw_img = raw_img.astype('float32')
nor_img = np.concatenate(normed_file_list, axis = 0)
nor_img = nor_img.astype('float32')

if raw_img.shape == nor_img.shape and nor_img.dtype == raw_img.dtype:
    ds = tf.data.Dataset.from_tensor_slices((raw_img, nor_img))
else:
    print("Shape of source img: ", raw_img.shape)
    print("Shape of target img: ", nor_img.shape)
    print("Type of source img: ", raw_img.dtype)
    print("Type of target img: ", nor_img.dtype)
    sys.exit("\033[93m  The size or type of source and target unmatched, check the size again \033[00m")

# Flush out unnecessary memory usage
del imag_array, padded_imag_array, img, raw_file_list, normed_file_list, desired_file_list

generator = tf.keras.models.load_model('./gen.h5')
generator.compile(optimizer = tf.keras.optimizers.Adam(1e-2, beta_1=0.5), loss = 'mse', metrics = [tf.keras.metrics.CosineSimilarity()])
predicted = generator.predict(ds.take(10).batch(5))

for f in range(10):
    os.mkdir(f"./oneStepNorm/testResult{f+1}")()
    for i in range(128):
        plt.subplot(1,3,1)
        plt.imshow(predicted[f][:,:,i,0], vmin= nor_img[f].min(),vmax = nor_img[f].max(),cmap = 'gray')
        plt.title('Predicted_img')
        plt.colorbar(shrink=0.5)
        plt.axis('off')

        plt.subplot(1,3,2)
        plt.imshow(nor_img[f][:,:,i,0], vmin= nor_img[f].min(),vmax = nor_img[f].max(), cmap = 'gray')
        plt.title('Normalised_img')
        plt.colorbar(shrink=0.5)
        plt.axis('off')

        diff = (predicted[f][:,:,i,0] - nor_img[f][:,:,i,0])
        plt.subplot(1,3,3)
        plt.imshow(diff, cmap = 'gray')
        plt.title('Difference')
        plt.axis('off')
        plt.colorbar(shrink=0.5)
        plt.savefig(os.path.join(f'./oneStepNorm/testResult{f+1}',f'Result_slice_{i+1}.png'))
        plt.close()

Report = {}
Report_loss = []
Report_cos = []
Report_ssim = []
for i in range(10):
    src_img = raw_img[i]
    src_img = src_img[None,...]
    tar_img = nor_img[i]
    tar_img = tar_img[None,...]
    predicted = generator.evaluate(src_img, tar_img)
    print(predicted)
    ssim = tf.image.ssim(src_img[:,:,:,:,0], tar_img[:,:,:,:,0], src_img[:,:,:,:,0].max() )
    print(ssim.numpy())
    Report_loss.append(predicted[0])
    Report_cos.append(predicted[1])
    Report_ssim.append(float(ssim.numpy()))
Report['MSE'] = Report_loss
Report["Cos"] = Report_cos
Report["SSIM"]= Report_ssim
df = pd.DataFrame(Report)
df.to_csv(f'{date}_Report')