
#from functions.Basic_tool import bn_relu_block
import sys, os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, 
    Dropout, 
    Activation, 
    BatchNormalization, 
    Conv3D, 
    Conv3DTranspose,
    AveragePooling3D,
    concatenate
    )
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l1_l2


def conv3d_bn_relu_block(input_tensor,
                         filter_nums, 
                         kernel_size, 
                         init='he_normal', 
                         param=0, 
                         batnorm=True):
    """
    input_tensors: a 5-d array or tensor as input
    filter_nums: num of filters for 
    kernel_size: self-defined kernel size for convolution
    param: stats for l2 kernel regulariser, default is 0, 
    batnorm: do batchnorm or not, default is True
    """
    y = Conv3D(filter_nums, 
               kernel_size, 
               padding='same', 
               use_bias=True,
               kernel_initializer = init,
               kernel_regularizer=l1_l2(l1=0,l2=param), 
               bias_regularizer=l1_l2(l1=0,l2=param))(input_tensor)
    
    if batnorm == True:
        y = BatchNormalization()(y)
    else:
        pass
    act = Activation('relu')(y)
    return act


def downSample_U_Net_block(input_tensor, filter_num, 
                           kernel_size, addDropout=False):

    conv1 = conv3d_bn_relu_block(input_tensor=input_tensor, 
                                 filter_nums=filter_num, 
                                 kernel_size=kernel_size)

    conv1 = conv3d_bn_relu_block(input_tensor=conv1, 
                                 filter_nums=filter_num, 
                                 kernel_size=kernel_size)

    if addDropout == True:
        conv1 = Dropout(0.5)(conv1)
    else:
        pass
    downSample = Conv3D(filter_num, 
                        kernel_size, 
                        strides=(2,2,2), 
                        padding='same'
                        )(conv1)
    return conv1, downSample

def upSample_U_Net_block(input_tensor, input_tensor_2,
                         filter_num, kernel_size):
    
    upSample = Conv3DTranspose(filter_num, 
                               kernel_size, 
                               strides=(2,2,2), 
                               padding='same'
                               )(input_tensor)

    concated_tensor = concatenate([upSample, input_tensor_2])

    conv = conv3d_bn_relu_block(input_tensor=concated_tensor, 
                                filter_nums=filter_num, 
                                kernel_size=kernel_size)
    
    conv = conv3d_bn_relu_block(input_tensor=conv, 
                                filter_nums=filter_num, 
                                kernel_size=kernel_size)

    return conv



class UNet_builder:
    
    # Make sure the data is in right shape and the checking functions is in global scope 
    def __init__(self) -> None:
        super().__init__()


    def _DataHandler3D(input_size):
        """
        Handling the size of input data
        """
        assert len(input_size) == 4, "Check the size of your input, a 3D input shall have dimenson 4"
    

    def build_U_Net3D(input_size, 
                      filter_num, 
                      kernel_size, 
                      pretrained_weights=None, 
                      trainable=True):
        """
        The input ndim must eqaul to 4 

        simplified the code with different pieces of functions,
        instead of series of  code like this 
        """
        UNet_builder._DataHandler3D(input_size) 

        input1 = Input(input_size)
        
        conv1, downSampleB1 = downSample_U_Net_block(input_tensor=input1, filter_num=filter_num, 
                                                     kernel_size=kernel_size)
        _, downSampleB2 = downSample_U_Net_block(input_tensor=downSampleB1, filter_num=filter_num * 2, 
                                                 kernel_size=kernel_size)
        _, downSampleB3 = downSample_U_Net_block(input_tensor=downSampleB2, filter_num=filter_num * 4, 
                                                 kernel_size=kernel_size)

        _, downSampleB4 = downSample_U_Net_block(input_tensor=downSampleB3, filter_num=filter_num * 8, 
                                                 kernel_size=kernel_size, addDropout=True)
        _, downSampleB5 = downSample_U_Net_block(input_tensor=downSampleB4, filter_num=filter_num * 8, 
                                                 kernel_size=kernel_size)

        # Shall have greatest feature map?
        latent_space = Conv3D(filters=filter_num, kernel_size=(3,3,3), 
                              activation='relu', padding='same', 
                              kernel_initializer='he_normal'
                              )(downSampleB5)

        upSampleB0 = Conv3D(filters=filter_num, kernel_size=(3,3,3), 
                            activation='relu', padding='same', 
                            kernel_initializer='he_normal'
                            )(latent_space)

        upSampleB1 = upSample_U_Net_block(input_tensor=upSampleB0, input_tensor_2=downSampleB4, 
                                          filter_num=filter_num * 8, kernel_size=kernel_size)
        upSampleB2 = upSample_U_Net_block(input_tensor=upSampleB1, input_tensor=downSampleB3, 
                                          filter_num=filter_num * 8, kernel_size=kernel_size)
        upSampleB3 = upSample_U_Net_block(input_tensor=upSampleB2, input_tensor_2=downSampleB2, 
                                          filter_num=filter_num * 4, kernel_size=kernel_size)
        upSampleB4 = upSample_U_Net_block(input_tensor=upSampleB3, input_tensor_2=downSampleB1, 
                                          filter_num=filter_num * 2, kernel_size=kernel_size)
        upSampleB5 = upSample_U_Net_block(input_tensor=upSampleB4, input_tensor_2=conv1, 
                                          filter_num=filter_num, kernel_size=kernel_size)

        # Final Output, shall not be with activation function?
        conv10 = Conv3D(filters=1, kernel_size=1)(upSampleB5)

        model = Model(inputs=input1, outputs=conv10)


        if(pretrained_weights):
            model.load_weights(pretrained_weights)

        model.trainable = trainable
        return model 