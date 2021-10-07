import os 
import argparse
import tensorflow as tf
from GAN import GAN
from models_builder import UNet_builder
from discriminator import toy_discriminator

"""
args:
input_shape = (128, 128, 128, 1)
dic_input_shape = (128, 128, 128, 2)
init_fil_Gen = 40
init_fil_Dis = 20
kernel_initialiser_dis = tf.random_normal_initializer(0., 0.02)
kernel_size = (3,3,3)
_LAMBDA = 100

# Set training hyper_parameters
_batchSize = 1
_epochs = 300
d_lr = 2e-6
g_lr = 2e-5
"""

def main():
    parser = argparse.ArgumentParser()
    # Add '--image_folder' argument using add_argument() including a help. The type is string (by default):
    parser.add_argument("--image_folder", type=str, default="./Data_raw")
    parser.add_argument('--log_output_path', type=str, default=None)
    parser.add_argument('--checkpoint_folder', type=str, default=None)
    parser.add_argument('--g_lr', type=float, default=2e-5)
    parser.add_argument('--d_lr', type=float, default=2e-6)
    parser.add_argument('--lambda', type=int, default=100)
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--input_shape', type=tuple, default=None)
    parser.add_argument('--disc_input_shape', type=tuple, default=None)
    parser.add_argument('--disc_kernel_initialiser', type=tuple, default=tf.random_normal_initializer(0., 0.02))
    
    # Parse the argument and store it in a dictionary:
    args = vars(parser.parse_args())
    print(args)
    
    # The tuple may have some issue, find a way to deal with it
    generator = UNet_builder.build_U_Net3D((64, 64, 64, 1), 10, (3, 3, 3))

    discriminator = toy_discriminator.build_toy_discriminator(input_shape= (64, 64, 64, 1), 
                                                            init_filter_nums=10, 
                                                            init_kernel_size=(3,3,3), 
                                                            kernel_init=args["disc_kernel_initialiser"], 
                                                            repetitions=1)
    # Linked the generator and discriminator to create a pix2pix model
    pix2pix = GAN(discriminator=discriminator, generator=generator)

    # Set up optimisers 
    generator_optimizer = tf.keras.optimizers.Adam(args["g_lr"], beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(args["d_lr"], beta_1=0.5)


    # Compile the model 
    pix2pix.compile(g_optimizer= generator_optimizer, 
                    d_optimizer = discriminator_optimizer,
                    loss_fn= [generator_loss, discriminator_loss])

    # Fit the model 
    hist = pix2pix.fit(ds.skip(10).map(tf_random_rotate_image, num_parallel_calls=32)
                    .shuffle(50).batch(args["batchsize"]), 
                    epochs=args["epochs"],
                    callbacks=None,
                    verbose=1)
    
    print(generator.summary())
    print(args["disc_kernel_initialiser"])
if __name__ == "__main__":
    main()