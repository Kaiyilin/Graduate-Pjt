 # Multiple Inputs
import os, scipy, sys, cv2, datetime, sklearn, random
import pandas as pd
import nibabel as nib
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Activation, BatchNormalization, Multiply
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Dropout, AveragePooling3D, GlobalAveragePooling3D, GlobalMaxPooling3D
from tensorflow.keras.metrics import Accuracy, Precision, Recall
from tensorflow.keras.layers import concatenate, add

# Data Preprocessing
def data_preprocessing(image):
    image = (image - image.min())/(image.max() - image.min()) 
    return image

def standardised(image): 
    img = (image - image.mean())/image.std() 
    return img 

# Read files
def myreadfile(dirr, norm=True):
    """
    This version can import 3D array regardless of the size
    """
    #cwd = os.getcwd()
    number = 0
    array_list = []
    imgs_array = np.array([])
    path_list=[f for f in os.listdir(dirr) if not f.startswith('.')]
    path_list.sort() #對讀取的路徑進行排序
    #print(path_list)
    for file in path_list:
        if file.endswith(".nii"):
            img = nib.load(os.path.join(dirr, file))
            img_array = img.get_fdata()
            if norm == True:
                img_array = data_preprocessing(img_array)
            else:
                pass
            img_array = img_array[None,...]
            array_list.append(img_array)
        else:
            pass 
    imgs_array = np.concatenate(array_list, axis=0)

    return number, imgs_array, path_list

def padding_zeros(array, pad_size, channel_last=True):
    
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

def model_structure(model):
    """
    Visualise model's architecture
    display feature map shapes
    """
    for i in range(len(model.layers)):
      layer = model.layers[i]
    # summarize output shape
      print(i, layer.name, layer.output.shape)

# tensorboard log directiory
logdir="./logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# model checkpoint
checkpoint_dir ="./trckpt/"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_prefix = os.path.join(checkpoint_dir, "weights-{epoch:02d}.hdf5")

print(checkpoint_dir)
print(logdir)
print("\ntf.__version__ is", tf.__version__)
print("\ntf.keras.__version__ is:", tf.keras.__version__)
print('\nImport completed')