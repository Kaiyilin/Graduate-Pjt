
#from functions.Basic_tool import bn_relu_block
import sys, os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Activation, BatchNormalization, Multiply
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Dropout, UpSampling3D, concatenate, Conv3DTranspose, AveragePooling3D
from tensorflow.keras.regularizers import l1_l2
import tensorflow.keras.backend as K
#import functions.Basic_tool as bst



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
    
    @staticmethod
    def build_UNet3D(input_size, filter_num, kernel_size, pretrained_weights= None, trainable = True):
        """
        The input ndim must eqaul to 4 
        """
        UNet_builder._DataHandler3D(input_size) 

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


        conv10 = Conv3D(1, 1, activation = 'sigmoid')(upSampleB5)

        model = Model(inputs = input1, outputs = conv10)

        #model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
        
        #model.summary()

        if(pretrained_weights):
            model.load_weights(pretrained_weights)

        model.trainable = trainable
        return model 

    ### give a shot on putting dense block into U_Net
    @staticmethod
    def buid_Dense_U_Net3D(input_shape , init_filter_num=32, up_init_filter_num=32, up_kernel_size = (3,3,3), growth_rate = 16, pretrained_weights= None, trainable = True):
        """
        input_shape: The input ndim must eqaul to 4 otherwise the system would exit the code, 
        init_filter_num: the number of extracting from, 
        growth_rate: how may features added per convolution,
        up_kernel_size: sele, 
        pretrained_weights: Have Pretraiened weights? default is None, 
        trainable = True"""
        # Pre-defined building block
        def bn_conv_swish(x,filters,kernel=(1,1,1),strides=1):
            
            x = BatchNormalization()(x)
            x = Conv3D(filters, kernel, strides=strides,padding = 'same', kernel_initializer='he_normal', use_bias = False)(x)
            x = Activation('swish')(x)
            return x
        
        def dense_block(x, filters, growth_rate, repetition):
            
            for _ in range(repetition):
                y = bn_conv_swish(x, filters)
                y = bn_conv_swish(y, growth_rate, (3,3,3))
                x = tf.keras.layers.concatenate([y,x])
            return x
    
        def transition_layer(x):
            x = bn_conv_swish(x, K.int_shape(x)[-1] //2 )
            x = AveragePooling3D((2,2,2), strides = 2, padding = 'same')(x)
            return x


        def upSample_DenseUnet_block(inputTensor, inputTensor_2 ,filter_num, kernel_size):
            upSample = Conv3DTranspose(filter_num, 
                                       kernel_size, 
                                       strides=(2,2,2), 
                                       padding = 'same',
                                       kernel_initializer= 'he_normal',
                                       use_bias = False)(inputTensor)
            upSample = Activation("swish")(upSample)
            concated_tensor = concatenate([upSample, inputTensor_2])
            conv = bn_conv_swish(concated_tensor, filters = filter_num, kernel=kernel_size)
            conv = bn_conv_swish(conv, filters = filter_num, kernel=kernel_size)
            return conv

        UNet_builder._DataHandler3D(input_shape) 

        input = Input(input_shape)
        conv1 = Conv3D(32, (8,8,8), padding = 'same', strides=2)(input) # (N, 64,64,64,32)
        conv2 = Conv3D(64, (3,3,3), padding = 'same', strides=2)(conv1) # (N, 32,32,32,32)
        dense1 = dense_block(conv2, init_filter_num, growth_rate, 3) # (None, 32, 32, 32, 96+k*3)
        trans1 = transition_layer(dense1)
        dense2 = dense_block(trans1, init_filter_num, growth_rate, 4)# (None, 16, 16, 16, 96+k*4)
        trans2 = transition_layer(dense2)
        dense3 = dense_block(trans2, init_filter_num, growth_rate, 12) # (None, 8, 8, 8, 96+k*12)
        trans3 = transition_layer(dense3)
        dense4 = dense_block(trans3, init_filter_num, growth_rate, 8) # (None, 4, 4, 4, 96+k*8)
        convTrans1 = upSample_DenseUnet_block(dense4, dense3, filter_num= up_init_filter_num, kernel_size=up_kernel_size)
        convTrans2 = upSample_DenseUnet_block(convTrans1, dense2, filter_num= up_init_filter_num, kernel_size=up_kernel_size)
        convTrans3 = upSample_DenseUnet_block(convTrans2, concatenate([dense1, conv2]),filter_num= up_init_filter_num, kernel_size=up_kernel_size)
        convTrans4 = upSample_DenseUnet_block(convTrans3, conv1, filter_num= up_init_filter_num, kernel_size=up_kernel_size)
        convTrans5 = Conv3DTranspose(32, up_kernel_size, 
                                    strides = 2, 
                                    padding = 'same', 
                                    kernel_initializer = 'he_normal',
                                    use_bias = False)(convTrans4)

        Act = Activation("swish")(convTrans5)
        lastConv = Conv3D(1,1, padding='same', use_bias=False)(Act)
        lastAct = Activation("swish")(lastConv)

        model = Model(inputs = input, outputs = lastAct)


        if(pretrained_weights):
            model.load_weights(pretrained_weights)

        model.trainable = trainable
        return model 

# --------------------------- cycleGAN class -----------------------------
class cycleGAN_3D(tf.keras.Model):
    """
    suggest you have 2 image domain X and Y
    generatorG convert X to Y
    generatorF convert Y to X

    discriminator_x classify the real and fake in X domain
    discriminator_y classify the real and fake in Y domain
    """
    def __init__(self, discriminator_x, discriminator_y, generatorG, generatorF):
        super(cycleGAN_3D, self).__init__()
        self.discriminator_x = discriminator_x
        self.discriminator_y = discriminator_y
        self.generatorG = generatorG      
        self.generatorF = generatorF

    def compile(self, d_optimizer_x, d_optimizer_y, g_optimizerG, g_optimizerF, loss_fn: list):
        super(cycleGAN_3D, self).compile()
        self.d_optimizer_x = d_optimizer_x
        self.d_optimizer_y = d_optimizer_y
        self.g_optimizerG = g_optimizerG
        self.g_optimizerF = g_optimizerF
        ### 4 kind of loss 
        self.generator_loss = loss_fn[0]
        self.discriminator_loss = loss_fn[1]
        self.cycle_loss = loss_fn[2]
        self.identity_loss = loss_fn[3]

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
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
        real_x, real_y = data
        with tf.GradientTape(persistent=True) as tape:
        # Generator G translates X -> Y
        # Generator F translates Y -> X.

            fake_y = self.generatorG(real_x, training=True)
            cycled_x = self.generatorF(fake_y, training=True)

            fake_x = self.generatorF(real_y, training=True)
            cycled_y = self.generatorG(fake_x, training=True)

            # same_x and same_y are used for identity loss.
            same_x = self.generatorF(real_x, training=True)
            same_y = self.generatorG(real_y, training=True)

            disc_real_x = self.discriminator_x(real_x, training=True)
            disc_real_y = self.discriminator_y(real_y, training=True)

            disc_fake_x = self.discriminator_x(fake_x, training=True)
            disc_fake_y = self.discriminator_y(fake_y, training=True)

            # calculate the loss
            gen_g_loss = self.generator_loss(disc_fake_y)
            gen_f_loss = self.generator_loss(disc_fake_x)

            total_cycle_loss = self.cycle_loss(real_x, cycled_x) + self.cycle_loss(real_y, cycled_y)

            # Total generator loss = adversarial loss + cycle loss
            total_gen_g_loss = gen_g_loss + total_cycle_loss + self.identity_loss(real_y, same_y)
            total_gen_f_loss = gen_f_loss + total_cycle_loss + self.identity_loss(real_x, same_x)

            disc_x_loss = self.discriminator_loss(disc_real_x, disc_fake_x)
            disc_y_loss = self.discriminator_loss(disc_real_y, disc_fake_y)

        # Calculate the gradients for generator and discriminator
        generatorG_gradients = tape.gradient(total_gen_g_loss, 
                                            self.generatorG.trainable_variables)
        generatorF_gradients = tape.gradient(total_gen_f_loss, 
                                            self.generatorF.trainable_variables)

        discriminator_x_gradients = tape.gradient(disc_x_loss, 
                                                self.discriminator_x.trainable_variables)
        discriminator_y_gradients = tape.gradient(disc_y_loss, 
                                                self.discriminator_y.trainable_variables)

        # Apply the gradients to the optimizer
        self.g_optimizerG.apply_gradients(zip(generatorG_gradients, 
                                                self.generatorG.trainable_variables))

        self.g_optimizerF.apply_gradients(zip(generatorF_gradients, 
                                                self.generatorF.trainable_variables))

        self.d_optimizer_x.apply_gradients(zip(discriminator_x_gradients,
                                                    self.discriminator_x.trainable_variables))

        self.d_optimizer_y.apply_gradients(zip(discriminator_y_gradients,
                                                    self.discriminator_y.trainable_variables))

        return {"total_gen_g_loss": total_gen_g_loss, "total_gen_f_loss": total_gen_f_loss, "disc_x_loss": disc_x_loss, "disc_y_loss": disc_y_loss}

    def summary_all(self):
        """
        Giving the summary of the cycleGAN model
        """
        print(self.generatorG.summary())
        print(self.generatorF.summary())
        print(self.discriminator_x.summary())
        print(self.discriminator_y.summary())

    def save_all(self, save_path: str, genG_name: str, genF_name: str, disc_x_name: str, disc_y_name: str, mode = 0):
        """
        save_path: str, given a save path 
        genG_name: str, given a name for generatorG 
        genF_name: str, given a name for generatorF
        disc_x_name: str, given a name for discriminator_x 
        disc_y_name: str, given a name for discriminator_y
        mode: 0 or 1, default 0 for save all the architectures, weights and bias
              1 for saving weights and bias only.
        """
        try:
            os.mkdir(save_path)
        except:
            pass
        
        if mode == 0:
            self.generatorG.save(os.path.join(save_path, f"{genG_name}.h5"))
            self.generatorF.save(os.path.join(save_path, f"{genF_name}.h5"))
            self.discriminator_x.save(os.path.join(save_path, f"{disc_x_name}.h5"))
            self.discriminator_y.save(os.path.join(save_path, f"{disc_y_name}.h5"))
        elif mode ==1:
            self.generatorG.save_weights(os.path.join(save_path, f"weights_{genG_name}.h5"))
            self.generatorF.save_weights(os.path.join(save_path, f"weights_{genF_name}.h5"))
            self.discriminator_x.save_weights(os.path.join(save_path, f"weights_{disc_x_name}.h5"))
            self.discriminator_y.save_weights(os.path.join(save_path, f"weights_{disc_y_name}.h5"))
        else:
            sys.exit('The mode should be chosen in either 0 or 1 if you truly like to save models')
    
    def save_separately(self, save_path: str, genG_name: str, genF_name: str, disc_x_name: str, disc_y_name: str):
        """
        save_path: str, given a save path 
        genG_name: str, given a name for generatorG 
        genF_name: str, given a name for generatorF
        disc_x_name: str, given a name for discriminator_x 
        disc_y_name: str, given a name for discriminator_y
        mode: 0 or 1, default 0 for save all the architectures, weights and bias
              1 for saving weights and bias only.
        """
        # Not finished yet