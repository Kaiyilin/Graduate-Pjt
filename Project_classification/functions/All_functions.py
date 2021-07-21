 # Multiple Inputs
import os
import sys
#import cv2
import h5py
import datetime
import sklearn
import random
import pandas as pd
import nibabel as nib
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l2, l1, l1_l2 # By using both, you can implemented the concept of elastic dense net
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Activation, BatchNormalization, Multiply
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Dropout, AveragePooling3D, GlobalAveragePooling3D, GlobalMaxPooling3D
from tensorflow.keras.metrics import Accuracy, Precision, Recall
from tensorflow.keras.layers import concatenate, add
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold #use for cross validation
from sklearn import preprocessing
#mport keras.backend.tensorflow_backend as tfback
import scipy
print('\nImport completed')



#metric_info = [tf.keras.metrics.SparseCategoricalAccuracy(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
#optimisers
#opt = tf.keras.optimizers.Adam(lr=0.001)
#opt2 = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9)
#opt3 = tf.keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)

#losses
#loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

# set the dictionary for files
dir= {
    'BA':'',
    'BA2':'',
    'BB':'',
    'BB2':'',
    'HC':'',
    'HC2':''
    }


# Data Preprocessing
def data_preprocessing(image):
    image = (image - image.min())/(image.max() - image.min()) 
    return image

def standardised(image): 
    img = (image - image.mean())/image.std() 
    return img 

# Read files
def myreadfile(dirr):
    """
    This version can import 3D array regardless of the size
    """
    os.chdir(dirr)
    #cwd = os.getcwd()

    number = 0

    flag = True
    imgs_array = np.array([])
    path_list=[f for f in os.listdir(dirr) if not f.startswith('.')]
    path_list.sort() #對讀取的路徑進行排序
    for file in path_list:
          if file.endswith(".nii"):
            #print(os.path.join(dirr, file))
            img = nib.load(os.path.join(dirr, file))
            img_array = img.get_fdata()
            img_array = data_preprocessing(img_array)
            img_array = img_array.reshape(-1,img_array.shape[0],img_array.shape[1],img_array.shape[2])
            number += 1
            if flag == True:
                imgs_array = img_array

            else:
                imgs_array = np.concatenate((imgs_array, img_array), axis=0)

            flag = False
    return number, imgs_array, path_list

def padding_zeros(array, pad_size):
    # define padding size
    elements = array.shape    
    for element in elements:
        if element > pad_size:
            sys.exit('\nThe expanded dimension shall be greater than your current dimension')
    pad_list = list() 
    for i in range(array.ndim):
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
    pad_array = np.pad(array, pad_list, 'constant')
    pad_list = list() 
    return pad_array

def myreadfile_pad(dirr, pad_size):
    
    #This version can import 3D array regardless of the size
    
    os.chdir(dirr)
    number = 0

    flag = True
    imgs_array = np.array([])
    path_list=[f for f in os.listdir(dirr) if not f.startswith('.')]
    path_list.sort()
    for file in path_list:
        if file.endswith(".nii"):
            #print(os.path.join(dirr, file))
            img = nib.load(os.path.join(dirr, file))
            img_array = img.get_fdata()
            img_array = data_preprocessing(img_array)
            img_array = padding_zeros(img_array, pad_size)
            img_array = img_array.reshape(-1,img_array.shape[0],img_array.shape[1],img_array.shape[2])
            number += 1
            if flag == True:
                imgs_array = img_array

            else:
                imgs_array = np.concatenate((imgs_array, img_array), axis=0)

            flag = False
    return number, imgs_array, path_list

def myreadfile_resample_pad(dirr, pad_size):
    #This version can import 3D array regardless of the size
    from nilearn.datasets import load_mni152_template
    from nilearn.image import resample_to_img
    template = load_mni152_template()

    os.chdir(dirr)
    number = 0

    flag = True
    imgs_array = np.array([])
    path_list=[f for f in os.listdir(dirr) if not f.startswith('.')]
    path_list.sort()
    for file in path_list:
        if file.endswith(".nii"):
            #print(os.path.join(dirr, file))
            img = nib.load(os.path.join(dirr, file))
            img_array = resample_to_img(img, template)
            img_array = img.get_fdata()
            img_array = data_preprocessing(img_array)
            img_array = padding_zeros(img_array, pad_size)
            img_array = img_array.reshape(-1,img_array.shape[0],img_array.shape[1],img_array.shape[2])
            number += 1
            if flag == True:
                imgs_array = img_array

            else:
                imgs_array = np.concatenate((imgs_array, img_array), axis=0)

            flag = False
    return number, imgs_array, path_list

def importdata_resample(dirr,dirr1,dirr2,dirr3,dirr4,dirr5,pad_size=None):
    def myreadfile_resample_pad(dirr, pad_size):
        #This version can import 3D array regardless of the size
        from nilearn.datasets import load_mni152_template
        from nilearn.image import resample_to_img
        template = load_mni152_template()

        os.chdir(dirr)
        number = 0

        flag = True
        imgs_array = np.array([])
        path_list=[f for f in os.listdir(dirr) if not f.startswith('.')]
        path_list.sort()
        for file in path_list:
            if file.endswith(".nii"):
                #print(os.path.join(dirr, file))
                img = nib.load(os.path.join(dirr, file))
                img_array = resample_to_img(img, template)
                img_array = img.get_fdata()
                img_array = data_preprocessing(img_array)
                img_array = padding_zeros(img_array, pad_size)
                img_array = img_array.reshape(-1,img_array.shape[0],img_array.shape[1],img_array.shape[2])
                number += 1
                if flag == True:
                    imgs_array = img_array

                else:
                    imgs_array = np.concatenate((imgs_array, img_array), axis=0)

                flag = False
        return number, imgs_array, path_list
    if pad_size == None:
      _, first_mo,  = myreadfile(dirr)
      _, second_mo, _ = myreadfile(dirr1)
      _, third_mo, _ = myreadfile(dirr2)
      
      _, first_mo2, _ = myreadfile(dirr3)
      _, second_mo2, _ = myreadfile(dirr4)
      _, third_mo2, _ = myreadfile(dirr5)
      print(first_mo.shape, second_mo.shape, third_mo.shape, first_mo2.shape, second_mo2.shape, third_mo2.shape)
      return first_mo, second_mo, third_mo, first_mo2, second_mo2, third_mo2

    else:
      _, first_mo, _ = myreadfile_resample_pad(dirr,pad_size)
      _, second_mo, _ = myreadfile_resample_pad(dirr1,pad_size)
      _, third_mo, _ = myreadfile_resample_pad(dirr2,pad_size)
      
      _, first_mo2, _ = myreadfile_resample_pad(dirr3,pad_size)
      _, second_mo2, _ = myreadfile_resample_pad(dirr4,pad_size)
      _, third_mo2, _ = myreadfile_resample_pad(dirr5,pad_size)
      print(first_mo.shape, second_mo.shape, third_mo.shape, first_mo2.shape, second_mo2.shape, third_mo2.shape)
      return first_mo, second_mo, third_mo, first_mo2, second_mo2, third_mo2

def importdata(dirr,dirr1,dirr2,dirr3,dirr4,dirr5):
    
    a_num, first_mo, _ = myreadfile(dirr)
    b_num, second_mo, _ = myreadfile(dirr1)
    h_num, third_mo, _ = myreadfile(dirr2)
    
    a_num2, first_mo2, _ = myreadfile(dirr3)
    b_num2, second_mo2, _ = myreadfile(dirr4)
    h_num2, third_mo2, _ = myreadfile(dirr5)
    print(first_mo.shape, second_mo.shape, third_mo.shape, first_mo2.shape, second_mo2.shape, third_mo2.shape)
    return first_mo, second_mo, third_mo, first_mo2, second_mo2, third_mo2

def importdata2(dirr,dirr1,dirr2,dirr3,dirr4,dirr5,pad_size=None):

    if pad_size == None:
      a_num, first_mo, _ = myreadfile(dirr)
      b_num, second_mo, _ = myreadfile(dirr1)
      h_num, third_mo, _ = myreadfile(dirr2)
      
      a_num2, first_mo2, _ = myreadfile(dirr3)
      b_num2, second_mo2, _ = myreadfile(dirr4)
      h_num2, third_mo2, _ = myreadfile(dirr5)
      print(first_mo.shape, second_mo.shape, third_mo.shape, first_mo2.shape, second_mo2.shape, third_mo2.shape)
      return first_mo, second_mo, third_mo, first_mo2, second_mo2, third_mo2

    else:
      #pad_size = int(input('Which size would you like? '))
      a_num, first_mo, _ = myreadfile_pad(dirr,pad_size)
      b_num, second_mo, _ = myreadfile_pad(dirr1,pad_size)
      h_num, third_mo, _ = myreadfile_pad(dirr2,pad_size)
      
      a_num2, first_mo2, _ = myreadfile_pad(dirr3,pad_size)
      b_num2, second_mo2, _ = myreadfile_pad(dirr4,pad_size)
      h_num2, third_mo2, _ = myreadfile_pad(dirr5,pad_size)
      print(first_mo.shape, second_mo.shape, third_mo.shape, first_mo2.shape, second_mo2.shape, third_mo2.shape)
      return first_mo, second_mo, third_mo, first_mo2, second_mo2, third_mo2

def importgft(filepath,i,j,k,l,m):
    """
    """
    All = pd.read_csv(filepath)
    X = All.iloc[i:j,k:l].values # switch to array
    if m==None:
      print(X.shape, X.ndim)
      return X
    else:
      X_label = All.iloc[i:j,m].values # switch to arraymod
      X_label = X_label.astype(float)
      print(X.shape, X.ndim)
      print(X_label.shape, X_label)
      return X, X_label

def importgft2(filepath,i,j):
    """
    Selecting Rows based on column label
    i : the name or index of the column
    j : the selecting rules
    """
    All = pd.read_csv(filepath)
    X = All.loc[All.loc[:,i]==j,:] # switch to array
    X_labels = X.loc[:,i].values
    X = X.values
    print(X.shape,X_labels.shape)
    return X, X_labels

def importgflist(dic,i,j,k,l):
    GF = np.array([])
    flag = True
    for element in dic:
      gf = importgft(gfdir[element],i,j,k,l,m=None)
      if flag == True:
        GF = gf
      else:
        GF = np.concatenate([GF, gf], axis=1)
      flag = False
    return GF


def split(c,array):
    array_val = array[:c,:,:,:]
    array_tr = array[c:,:,:,:]
    return array_tr, array_val

# Augmentation
def translateit(image, offset, isseg=False):
    order = 0 if isseg == True else 5

    return scipy.ndimage.interpolation.shift(image, (int(offset[0]), int(offset[1]), 0), order=order, mode='nearest')

def translateit2(image, offset, isseg=False):
    order = 0 if isseg == True else 5

    return scipy.ndimage.interpolation.shift(image, (int(offset[0]), int(offset[1]), int(offset[2])), order=order, mode='nearest')

def rotateit_y(image, theta, isseg=False):
    order = 0 if isseg == True else 5
        
    return scipy.ndimage.rotate(image, float(theta), axes=(0,2), reshape=False, order=order, mode='nearest') # Shall detemined reshape or not, also rotate toward which axis?

def rotateit_x(image, theta, isseg=False):
    order = 0 if isseg == True else 5
        
    return scipy.ndimage.rotate(image, float(theta), axes=(1,2), reshape=False, order=order, mode='nearest') # Shall detemined reshape or not, also rotate toward which axis?

def rotateit_z(image, theta, isseg=False):
    order = 0 if isseg == True else 5
        
    return scipy.ndimage.rotate(image, float(theta), axes=(0,1), reshape=False, order=order, mode='nearest') # Shall detemined reshape or not, also rotate toward which axis?

def aug_translation(datasets,offset_list):
    """
    datasets refer to those arrays you'd like to use
    offset_list refer to the list of distance you'd like to shift in (x,y)
    """
    num=0
    Flag = True
    array_aug = np.array([])
    for offset in offset_list: 
        for array in datasets:
            array_shift = translateit(array, offset)
            array_shift = array_shift.reshape(-1,array_shift.shape[0],array_shift.shape[1],array_shift.shape[2])
            num +=1
            if Flag == True:
                array_aug = array_shift
            else:
                array_aug = np.concatenate([array_aug,array_shift])
            Flag = False
    print('\nTranslation complete, the total amount of translated subjects are:'+'%s'%(num))
    return array_aug

def aug_rotate_x(datasets,theta_list):
    """
    datasets refer to those arrays you'd like to use
    theta_list refer to the list of rotating angle you'd like to shift
    """
    num=0
    Flag = True
    array_aug = np.array([])
    #output_dataset = np.array([])
    for theta in theta_list:       
        for array in datasets:
            array_rotate = rotateit_x(array, theta)
            array_rotate = array_rotate.reshape(-1,array_rotate.shape[0],array_rotate.shape[1],array_rotate.shape[2])
            num +=1
            if Flag == True:
                array_aug = array_rotate
            else:
                array_aug = np.concatenate([array_aug,array_rotate])
            Flag = False

    print('\nRotate augmentation complete, the total amount of rotated subjects are:'+'%s'%(num))
    return array_aug

def aug_rotate_y(datasets,theta_list):
    """
    datasets refer to those arrays you'd like to use
    theta_list refer to the list of rotating angle you'd like to shift
    """
    num=0
    Flag = True
    array_aug = np.array([])
    #output_dataset = np.array([])
    for theta in theta_list:       
        for array in datasets:
            array_rotate = rotateit_y(array, theta)
            array_rotate = array_rotate.reshape(-1,array_rotate.shape[0],array_rotate.shape[1],array_rotate.shape[2])
            num +=1
            if Flag == True:
                array_aug = array_rotate
            else:
                array_aug = np.concatenate([array_aug,array_rotate])
            Flag = False

    print('\nRotate augmentation complete, the total amount of rotated subjects are:'+'%s'%(num))
    return array_aug

def aug_rotate_z(datasets,theta_list):
    """
    datasets refer to those arrays you'd like to use
    theta_list refer to the list of rotating angle you'd like to shift
    """
    num=0
    Flag = True
    array_aug = np.array([])
    #output_dataset = np.array([])
    for theta in theta_list:       
        for array in datasets:
            array_rotate = rotateit_z(array, theta)
            array_rotate = array_rotate.reshape(-1,array_rotate.shape[0],array_rotate.shape[1],array_rotate.shape[2])
            num +=1
            if Flag == True:
                array_aug = array_rotate
            else:
                array_aug = np.concatenate([array_aug,array_rotate])
            Flag = False

    print('\nRotate augmentation complete, the total amount of rotated subjects are:'+'%s'%(num))
    return array_aug

def training_set_generator(datasets,offset_list,theta_list,aug=True):
    if aug == True:
        aug_datasets_trans = aug_translation(datasets,offset_list)
        aug_datasets_rotate_x = aug_rotate_x(aug_datasets_trans,theta_list)
        aug_datasets_rotate_y = aug_rotate_y(aug_datasets_trans,theta_list)
        aug_datasets_rotate_z = aug_rotate_z(aug_datasets_trans,theta_list)
        output_dataset = np.concatenate([datasets,aug_datasets_trans,aug_datasets_rotate_x,aug_datasets_rotate_y,aug_datasets_rotate_z],axis=0)
    else:
        output_dataset = datasets
        print('\nTraining set output without any modification')
    
    return output_dataset

# Basic conv block 
def bn_relu_block(input,filter,kernel_size,param):
    y = Conv3D(filter, kernel_size, padding='same', use_bias=True,kernel_initializer ='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param), bias_regularizer=l1_l2(l1=0,l2=param))(input)
    y = BatchNormalization()(y)
    act = Activation('relu')(y)
    tf.summary.histogram("Activation", y)# testing 
    return y

# Basic SE, inception and parametre reduction module

def half_reduction(input,filters):
    
    conv00 = Conv3D(filters,(1,1,1),padding='same')(input)
    conv00 = Conv3D(filters,(3,3,3),padding='same')(conv00)
    conv00 = Conv3D(filters, (3,3,3),strides=(2,2,2),padding='same')(conv00)

    conv01 = Conv3D(filters,(1,1,1),padding='same')(input)
    conv01 = Conv3D(filters, (3,3,3),strides=(2,2,2),padding='same')(conv01)
 
    avg00 = MaxPooling3D(pool_size=(2,2,2))(input)

    concatenate = tf.keras.layers.concatenate([conv00, conv01, avg00])
    concatenate = BatchNormalization()(concatenate)
    #y = Activation('relu')(y)
    return Activation('relu')(concatenate)

def inception_module3D(input_img,filters,param):
    incep_1 = Conv3D(filters, (1,1,1), padding='same', activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(input_img)
    incep_1 = Conv3D(filters, (3,3,3), padding='same', activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(incep_1)
    incep_2 = Conv3D(filters, (1,1,1), padding='same', activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(input_img)
    incep_2 = Conv3D(filters, (5,5,5), padding='same', activation='relu')(incep_2)
    incep_3 = MaxPooling3D((3,3,3), strides=(1,1,1), padding='same')(input_img)
    incep_3 = Conv3D(filters, (1,1,1), padding='same', activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(incep_3)
    output = tf.keras.layers.concatenate([incep_1, incep_2, incep_3], axis = 4)
    return output
    
def inc_module_A(input,filters,param):
  chan_1 = Conv3D(filters, (1,1,1), padding = 'same',activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(input)
  chan_1 = Conv3D(filters, (3,3,3), padding = 'same',activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_1)
  chan_1 = Conv3D(filters, (3,3,3), padding = 'same',activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_1)

  chan_2 = Conv3D(filters, (1,1,1),padding = 'same',activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(input)
  chan_2 = Conv3D(filters, (3,3,3),padding = 'same',activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_2)
  
  chan_3 = AveragePooling3D((3,3,3), strides=(1,1,1), padding='same')(input)
  chan_3 = Conv3D(filters, (1,1,1),padding = 'same',activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_3)
  
  chan_4 = Conv3D(filters, (1,1,1),activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(input)
  
  output = tf.keras.layers.concatenate([chan_1,chan_2,chan_3,chan_4],axis=4)
  return output

def inc_module_B(input,filters,param):

  chan_1 = Conv3D(filters, (1,1,1), padding = 'same',activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(input)
  chan_1 = Conv3D(filters, (7,1,1), padding = 'same',activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_1)
  chan_1 = Conv3D(filters, (1,7,7),padding = 'same',activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_1)
  chan_1 = Conv3D(filters, (7,1,1), padding = 'same',activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_1)
  chan_1 = Conv3D(filters, (1,7,7),padding = 'same',activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_1)
  
  chan_2 = Conv3D(filters, (1,1,1),padding = 'same',activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(input)
  chan_2 = Conv3D(filters, (7,1,1),padding = 'same',activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_2)
  chan_2 = Conv3D(filters, (1,7,7),padding = 'same',activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_2)
  
  chan_3 = AveragePooling3D((3,3,3), strides=(1,1,1), padding='same')(input)
  chan_3 = Conv3D(filters, (1,1,1),padding = 'same',activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_3)
  
  chan_4 = Conv3D(filters, (1,1,1),activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(input)
  
  output = tf.keras.layers.concatenate([chan_1,chan_2,chan_3,chan_4],axis=4)
  return output

def inc_module_C(input,filters,param):
  chan_1 = Conv3D(filters, (1,1,1), padding = 'same',activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(input)
  chan_1 = Conv3D(filters, (3,3,3), padding = 'same',activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_1)
  chan_12 = Conv3D(filters, (3,1,1), padding = 'same',activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_1)
  chan_13 = Conv3D(filters, (1,3,3), padding = 'same',activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_1)

  chan_2 = Conv3D(filters, (1,1,1),padding = 'same',activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(input)
  chan_21 = Conv3D(filters, (1,3,3),padding = 'same',activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_2)
  chan_22 = Conv3D(filters, (3,1,1),padding = 'same',activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_2)
  
  chan_3 = AveragePooling3D((3,3,3), strides=(1,1,1), padding='same')(input)
  chan_3 = Conv3D(filters, (1,1,1),padding = 'same',activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_3)
  
  chan_4 = Conv3D(filters, (1,1,1),activation='relu',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(input)
  
  output = tf.keras.layers.concatenate([chan_12, chan_13, chan_21, chan_22, chan_3, chan_4],axis=4)
  return output
    
def inc_module_A_2(input,filters,param):
  chan_1 = Conv3D(filters, (1,1,1), padding = 'same',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(input)
  chan_1 = Conv3D(filters, (3,3,3), padding = 'same',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_1)
  chan_1 = Conv3D(filters, (3,3,3), padding = 'same',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_1)

  chan_2 = Conv3D(filters, (1,1,1), padding = 'same',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(input)
  chan_2 = Conv3D(filters, (3,3,3), padding = 'same',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_2)
  
  chan_3 = AveragePooling3D((3,3,3), strides=(1,1,1), padding='same')(input)
  chan_3 = Conv3D(filters, (1,1,1), padding = 'same',kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_3)
  
  chan_4 = Conv3D(filters, (1,1,1), kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(input)
  
  output = tf.keras.layers.concatenate([chan_1,chan_2,chan_3,chan_4],axis=4)
  output = BatchNormalization()(output)
  output = Activation('relu')(output)
  return output

def inc_module_B_2(input,filters,param):

  chan_1 = Conv3D(filters, (1,1,1), padding = 'same', kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(input)
  chan_1 = Conv3D(filters, (7,1,1), padding = 'same', kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_1)
  chan_1 = Conv3D(filters, (1,7,7), padding = 'same', kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_1)
  chan_1 = Conv3D(filters, (7,1,1), padding = 'same', kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_1)
  chan_1 = Conv3D(filters, (1,7,7), padding = 'same', kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_1)


  chan_2 = Conv3D(filters, (1,1,1),padding = 'same', kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(input)
  chan_2 = Conv3D(filters, (7,1,1),padding = 'same', kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_2)
  chan_2 = Conv3D(filters, (1,7,7),padding = 'same', kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_2)
  
  
  chan_3 = AveragePooling3D((3,3,3), strides=(1,1,1), padding='same')(input)
  chan_3 = Conv3D(filters, (1,1,1), padding = 'same', kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_3)
  
  chan_4 = Conv3D(filters, (1,1,1), kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(input)
  
  output = tf.keras.layers.concatenate([chan_1,chan_2,chan_3,chan_4],axis=4)
  output = BatchNormalization()(output)
  output = Activation('relu')(output)
  return output

def inc_module_C_2(input,filters,param):
  chan_1 = Conv3D(filters, (1,1,1), padding = 'same', kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(input)
  chan_1 = Conv3D(filters, (3,3,3), padding = 'same', kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_1)
  chan_12 = Conv3D(filters, (3,1,1), padding = 'same', kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_1)
  chan_13 = Conv3D(filters, (1,3,3), padding = 'same', kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_1)

  chan_2 = Conv3D(filters, (1,1,1), padding = 'same', kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(input)
  chan_21 = Conv3D(filters, (1,3,3), padding = 'same', kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_2)
  chan_22 = Conv3D(filters, (3,1,1), padding = 'same', kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_2)
  
  chan_3 = AveragePooling3D((3,3,3), strides=(1,1,1), padding='same')(input)
  chan_3 = Conv3D(filters, (1,1,1), padding = 'same', kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(chan_3)
  
  chan_4 = Conv3D(filters, (1,1,1), kernel_initializer='he_normal',kernel_regularizer=l1_l2(l1=0,l2=param))(input)
  
  output = tf.keras.layers.concatenate([chan_12, chan_13, chan_21, chan_22, chan_3, chan_4],axis=-1)
  output = BatchNormalization()(output)
  output = Activation('relu')(output)
  return output

def se_block_3D(tensor, ratio):
    nb_channel = tensor.shape[-1] # for channel last

    x = GlobalAveragePooling3D()(tensor)
    x = Dense(nb_channel // ratio, activation='relu',use_bias=False)(x)
    x = Dense(nb_channel, activation='sigmoid')(x)

    x = tf.keras.layers.Multiply()([tensor, x])
    return x

def Conv_SE_block(input, filters, kernel_size, param, ratio, SE = True):
    """
    build a conv --> SE block
    """
    y = Conv3D(filters, 
               kernel_size, 
               padding='same',
               kernel_initializer='he_normal',
               kernel_regularizer=l1_l2(l1=0,l2=param), 
               bias_regularizer=l1_l2(l1=0,l2=0))(input)
    if SE == True:
        y = se_block_3D(y, ratio)
    else: 
        y=y
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    return y    


# model training

def base_model_creator(model, train_para = False):
    base_model = tf.keras.models.Model([model.inputs], [model.get_layer(index=-2).output])
    base_model.trainable = train_para
    return base_model

def model_structure(model):
    """
    Visualise model's architecture
    display feature map shapes
    """
    for i in range(len(model.layers)):
      layer = model.layers[i]
    # summarize output shape
      print(i, layer.name, layer.output.shape)

def decay(epoch):
  if epoch <= 30:
    return 1e-1
  elif epoch > 30 and epoch <= 70:
    return 1e-2
  else:
    return 1e-3

# Setting callbacks

# tensorboard log directiory
logdir="/home/kaiyi/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# model checkpoint
checkpoint_dir ="/home/kaiyi/trckpt/"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_prefix = os.path.join(checkpoint_dir, "weights-{epoch:02d}.hdf5")


class PrintLR(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    print('\nLearning rate for epoch {} is {}'.format(epoch + 1,
                                                      model.optimizer.lr.numpy()))



print(checkpoint_dir)
print(logdir)

print("\ntf.__version__ is", tf.__version__)
print("\ntf.keras.__version__ is:", tf.keras.__version__)