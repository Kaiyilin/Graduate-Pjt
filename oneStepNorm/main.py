import os
import time
import scipy
import argparse
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from configs import prjt_configs
from GAN import GAN
from generator import UNet_builder
from discriminator.ToyDiscriminator import ToyDiscriminator
from utils.augmentation import tf_random_rotate_image
from dataloader.dataloader import DataLoader
# from dataloader.dataloader import ds

LOG_NAME = "execution_record.log"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str, default=prjt_configs["records"]["log_path"])
    parser.add_argument('--ckpt_path', type=str, default=prjt_configs["records"]["ckpt_path"])
    parser.add_argument('--gen_input_shape', default=prjt_configs["model"]["gen_input_shape"])
    parser.add_argument('--disc_input_shape', default=prjt_configs["model"]["disc_input_shape"])
    parser.add_argument('--gen_filter_nums', type=int, default=prjt_configs["train"]["gen_filter_nums"])
    parser.add_argument('--disc_filter_nums', type=int, default=prjt_configs["train"]["disc_filter_nums"])
    parser.add_argument('--kernel_size', type=int, default=prjt_configs["train"]["kernel_size"])
    parser.add_argument('--g_lr', type=float, default=prjt_configs["train"]["g_lr"])
    parser.add_argument('--d_lr', type=float, default=prjt_configs["train"]["d_lr"])
    parser.add_argument('--lambda', type=int, default=prjt_configs["train"]["_lambda"])
    parser.add_argument('--batchsize', type=int, default=prjt_configs["train"]["_batchSize"])
    parser.add_argument('--epochs', type=int, default=prjt_configs["train"]["_epochs"])

    # Parse the argument and store it in a dictionary:
    # vars returns the __dict__ attribute of the given object.
    args = vars(parser.parse_args())

    try:
        os.makedirs(args["log_path"])
        os.makedirs(args["ckpt_path"])
    except FileExistsError as e:
        print(e)

    start_time = time.time()


    # Set a logger for info
    logging.basicConfig(
        filename = './execution_record.log',
        level = logging.WARNING,
        format = '%(filename)s %(message)s'
        )

    # Self-defined loss function for GAN
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    _lambda = args["lambda"]

    def generator_loss(disc_generated_output, gen_output, target):
        gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

        # mean absolute error
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

        total_gen_loss = gan_loss + (_lambda * l1_loss)

        return total_gen_loss, gan_loss, l1_loss

    def discriminator_loss(disc_real_output, disc_generated_output):
        real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

        generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss
    
    ds = DataLoader(
        raw_data_path = prjt_configs,
        paired_data_path= prjt_configs
    )

    # create generator and discriminator
    generator = UNet_builder.build_U_Net3D(input_size=args["gen_input_shape"], 
                                           filter_num=args["gen_filter_nums"], 
                                           kernel_size=args["kernel_size"])

    discriminator = ToyDiscriminator.build_toy_discriminator(input_shape=args["disc_input_shape"], 
                                                              init_filter_nums=args["disc_filter_nums"], 
                                                              init_kernel_size=args["kernel_size"], 
                                                              kernel_init=tf.random_normal_initializer(0., 0.02), 
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
    hist = pix2pix.fit(
        ds.skip(30).map(tf_random_rotate_image, num_parallel_calls=32)
        .shuffle(50).batch(args["batchsize"]), 
        epochs=args["epochs"],
        callbacks=None,
        verbose=1
        )

    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(os.path.join(args["log_path"], "historty.csv"))

    # save model
    pix2pix.save_all(save_path=args['ckpt_path'], gen_name="generator", disc_name="discriminator")
    
    # log records
    duration = (time.time() - start_time) / 60
    print("test_PR")
    logging.warning(f"""Log Path: {args["log_path"]}, Ckpt Path: {args["ckpt_path"]}, 
                    Training_Duration: {duration:.2f} mins, l1_LossLambda: {args["lambda"]}, 
                    initail filter number of Generator: {args["gen_filter_nums"]}, 
                    initail filter number of Discriminator: {args["disc_filter_nums"]},
                    training in epoch:{args["epochs"]}, batchSize: {args["batchsize"]}, 
                    with G_LR: {args["g_lr"]} and D_LR: {args["d_lr"]}.""")




if __name__ == "__main__":
    main()