
#from functions.Basic_tool import bn_relu_block
import sys, os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Activation, BatchNormalization, Multiply
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Dropout, UpSampling3D, concatenate, Conv3DTranspose
#from keras.layers.merge import concatenate, add
from tensorflow.keras.regularizers import l1_l2
#import functions.Basic_tool as bst

def conv3d_bn_relu_block(input_tensors,filter_nums, kernel_size, init = 'he_normal', param = 0, batnorm=True):
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
               bias_regularizer=l1_l2(l1=0,l2=param))(input_tensors)
    if batnorm == True:
        y = BatchNormalization()(y)
    else:
        pass
    act = Activation('relu')(y)
    return act


def downSample_U_Net_block(inputTensor, filter_num, kernel_size, addDropout=False):
    conv1 = conv3d_bn_relu_block(inputTensor, filter_nums=filter_num, kernel_size=kernel_size)
    conv1 = conv3d_bn_relu_block(conv1, filter_nums=filter_num, kernel_size=kernel_size)
    if addDropout == True:
        conv1 = Dropout(0.5)(conv1)
    else:
        pass
    downSample = Conv3D(filter_num, 
                        kernel_size, 
                        strides=(2,2,2), 
                        padding = 'same')(conv1)
    return conv1, downSample

def upSample_U_Net_block(inputTensor, inputTensor_2 ,filter_num, kernel_size):
    upSample = Conv3DTranspose(filter_num, 
                               kernel_size, 
                               strides=(2,2,2), 
                               padding = 'same')(inputTensor)

    concated_tensor = concatenate([upSample, inputTensor_2])
    conv = conv3d_bn_relu_block(concated_tensor, filter_nums=filter_num, kernel_size=kernel_size)
    conv = conv3d_bn_relu_block(conv, filter_nums=filter_num, kernel_size=kernel_size)

    return conv


class UNet_builder:
    
    # Make sure the data is in right shape and the checking functions is in global scope 
    def __init__(self) -> None:
        super().__init__()


    def _DataHandler3D(input_size):
        """
        Handling the size of input data
        """
        if len(input_size) == 4:
            print(" \nInput size correct " + "\n" + '----'*5)
        else:
            sys.exit(" Check the size of your input, a 3D input shall have dimenson 4")
    

    def build_U_Net3D(input_size, filter_num, kernel_size, pretrained_weights= None, trainable = True):
        """
        The input ndim must eqaul to 4 

        simplified the code with different pieces of functions,
        instead of series of  code like this 
        """
        UNet_builder._DataHandler3D(input_size) 

        input1 = Input(input_size)
        
        conv1, downSampleB1 = downSample_U_Net_block(input1, filter_num, kernel_size)
        _, downSampleB2 = downSample_U_Net_block(downSampleB1, filter_num*2, kernel_size)
        _, downSampleB3 = downSample_U_Net_block(downSampleB2, filter_num*4, kernel_size)
        _, downSampleB4 = downSample_U_Net_block(downSampleB3, filter_num*8, kernel_size, addDropout=True)
        _, downSampleB5 = downSample_U_Net_block(downSampleB4, filter_num*8, kernel_size)

        # Shall have greatest feature map?
        Latent_space = Conv3D(filter_num, (3,3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(downSampleB5)

        upSampleB0 = Conv3D(filter_num, (3,3,3),activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(Latent_space)

        upSampleB1 = upSample_U_Net_block(upSampleB0, downSampleB4, filter_num*8, kernel_size)
        upSampleB2 = upSample_U_Net_block(upSampleB1, downSampleB3, filter_num*8, kernel_size)
        upSampleB3 = upSample_U_Net_block(upSampleB2, downSampleB2, filter_num*4, kernel_size)
        upSampleB4 = upSample_U_Net_block(upSampleB3, downSampleB1, filter_num*2, kernel_size)
        upSampleB5 = upSample_U_Net_block(upSampleB4, conv1, filter_num, kernel_size)

        # Final Output, shall not be with activation function?
        conv10 = Conv3D(1, 1)(upSampleB5)

        model = Model(inputs = input1, outputs = conv10)

        #model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
        
        #model.summary()

        if(pretrained_weights):
            model.load_weights(pretrained_weights)

        model.trainable = trainable
        return model 


class GAN(tf.keras.Model):
    """
    actually it's pix2pix
    """
    def __init__(self, discriminator, generator):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator      

    def compile(self, d_optimizer, g_optimizer, loss_fn: list):
        super(GAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        ### define loss
        self.generator_loss = loss_fn[0]
        self.discriminator_loss = loss_fn[1]
    """
    # 1 - With the "Functional API", 
    # where you start from Input, you chain layer calls to specify the model's forward pass,
    # and finally you create your model from inputs and outputs

    # 2 - By subclassing the Model class: in that case, 
    # you should define your layers in __init__ and you should implement the model's forward pass in call.
    # Check the custom layer documentation 
    def call(self, inputs, training, mask):
        return super().call(inputs, training=training, mask=mask)

    # take the summary for showing all the trainable parameters
    def summary(self, line_length, positions, print_fn):
        return super().summary(line_length=line_length, positions=positions, print_fn=print_fn)
    """
    def train_step(self, data):
        input_image, target = data
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)

            real_input =tf.keras.layers.concatenate([input_image, target],axis=-1)
            disc_real_output = self.discriminator(real_input, training=True)
            fake_input =tf.keras.layers.concatenate([input_image, gen_output],axis=-1)
            disc_generated_output = self.discriminator(fake_input, training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator_loss(disc_generated_output, gen_output, target)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                                self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                     self.discriminator.trainable_variables)

        self.g_optimizer.apply_gradients(zip(generator_gradients,
                                             self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(discriminator_gradients,
                                             self.discriminator.trainable_variables))
        return {"gen_total_loss":gen_total_loss, "gen_gan_loss": gen_gan_loss, "gen_l1_loss": gen_l1_loss, 'disc_loss': disc_loss}

    def summary_all(self):
        """
        Giving the summary of the cycleGAN model
        """
        print(self.generator.summary())
        print(self.discriminator.summary())

    def save_all(self, save_path: str, gen_name: str, disc_name: str, mode = 0):
        """
        save_path: str, given a save path 
        genG_name: str, given a name for generatorG 
        genF_name: str, given a name for generatorF
        disc_x_name: str, given a name for discriminator_x 
        disc_y_name: str, given a name for discriminator_y
        mode: 0 or 1, default 0 for save all the architectures, weights and bias
              1 for saving weights and bias only.
        """
        if mode == 0:
            self.generator.save(os.path.join(save_path, f"{gen_name}.h5"))
            self.discriminator.save(os.path.join(save_path, f"{disc_name}.h5"))
        elif mode ==1:
            self.generator.save_weights(os.path.join(save_path, f"{gen_name}.h5"))
            self.discriminator.save_weights(os.path.join(save_path, f"{disc_name}.h5"))
        else:
            sys.exit('The mode should be chosen in either 0 or 1 if you truly like to save models')
    
        # Not finished yet

