import os, scipy
import argparse
import numpy as np
import tensorflow as tf
from GAN import GAN
from models_builder import UNet_builder
from discriminator import toy_discriminator

# Set a callback for the validate data 
class evolRecord(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    predict_img = self.model.generator.predict(ds.take(1).batch(1))
    plt.imshow(predict_img[0, :, :, 64, 0], cmap='gray')
    plt.axis('off')
    plt.savefig(os.path.join(bst.checkpoint_dir, "PrdctImg_Epoch{}.png".format(epoch + 1)), dpi=72)

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

# tf data mapping function for data augmentation
def tf_random_rotate_image(image, image2):
    def random_rotate_image(image):
        image = scipy.ndimage.rotate(image, np.random.uniform(-60, 60), reshape=False)
        return image
    #im, im2 = images[0], images[1]
    image_shape = image.shape
    image2_shape = image2.shape
    [image,] = tf.py_function(random_rotate_image, [image], [tf.float32])
    image.set_shape(image_shape)
    image2.set_shape(image2_shape)
    return (image,image2)



def main():
    parser = argparse.ArgumentParser()
    # Add '--image_folder' argument using add_argument() including a help. The type is string (by default):
    parser.add_argument("--image_folder", type=str, default="./Data_raw")
    parser.add_argument('--log_output_path', type=str, default=None)
    parser.add_argument('--checkpoint_folder', type=str, default=None)
    parser.add_argument('--gen_filter_nums', type=int, default=40)
    parser.add_argument('--disc_filter_nums', type=int, default=20)
    parser.add_argument('--kernel_size', type=int, default=3) 
    parser.add_argument('--g_lr', type=float, default=2e-5)
    parser.add_argument('--d_lr', type=float, default=2e-6)
    parser.add_argument('--lambda', type=int, default=100)
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--input_shape', type=tuple, default=None)
    parser.add_argument('--disc_input_shape', type=tuple, default=None)
    
    # Parse the argument and store it in a dictionary:
    args = vars(parser.parse_args())
    print(args)
    
    # Self-defined loss function for GAN
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    _lambda = args["lambda"]

    # The tuple may have some issue, find a way to deal with it
    generator = UNet_builder.build_U_Net3D(input_size=(64, 64, 64, 1), 
                                           filter_num=args["gen_filter_nums"], 
                                           kernel_size=args["kernel_size"])

    discriminator = toy_discriminator.build_toy_discriminator(input_shape=(64, 64, 64, 1), 
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
        ds.skip(10).map(tf_random_rotate_image, num_parallel_calls=32)
        .shuffle(50).batch(args["batchsize"]), 
        epochs=args["epochs"],
        callbacks=None,
        verbose=1
        )

    print(generator.summary())

if __name__ == "__main__":
    main()