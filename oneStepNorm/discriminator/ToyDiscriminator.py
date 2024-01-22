#Toy_discriminator
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, 
                                     Activation, 
                                     BatchNormalization, 
                                     Conv3D, 
                                     MaxPooling3D, 
                                     Dropout, 
                                     concatenate, 
                                     AveragePooling3D)
import tensorflow.keras.backend as K

class ToyDiscriminator(object):

    @staticmethod
    def bn_conv_act(x, 
                   filters, 
                   activation='swish', 
                   kernel=(1,1,1), 
                   kernel_init='he_normal', 
                   strides=1):
        
        x = BatchNormalization()(x)
        x = Conv3D(filters, 
                   kernel, 
                   strides=strides,
                   padding='same', 
                   kernel_initializer=kernel_init, 
                   use_bias=False)(x)
        x = Activation(activation)(x)

        return x


    def build_toy_discriminator(input_shape: tuple, 
                                init_filter_nums: int, 
                                init_kernel_size: tuple, 
                                kernel_init, 
                                repetitions: int):
        """ for every repetions, the size of image will shrink to half, 
        """
        inputs = Input(input_shape, name ='input_imgs')
        down = ToyDiscriminator.bn_conv_act(inputs, 
                                            filters=init_filter_nums, 
                                            kernel=init_kernel_size, 
                                            kernel_init=kernel_init,
                                            strides=2)
        for rep in range(repetitions):
            init_filter_nums *= 2 
            down = ToyDiscriminator.bn_conv_act(down, 
                                                 filters=init_filter_nums, 
                                                 kernel=init_kernel_size, 
                                                 kernel_init=kernel_init,
                                                 strides=2)

        pre_conv = ToyDiscriminator.bn_conv_act(down, 
                                                filters=init_filter_nums*2, 
                                                kernel=init_kernel_size, 
                                                kernel_init=kernel_init,
                                                strides=1)

        last_conv = Conv3D(filters=1, 
                          kernel_size=4, 
                          strides=1,
                          kernel_initializer=kernel_init, 
                          padding='same')(pre_conv)

        return tf.keras.Model(inputs=inputs, outputs=last_conv)