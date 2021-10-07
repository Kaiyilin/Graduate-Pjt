from GAN import GAN
import tensorflow as tf
from GAN import UNet_builder
from discriminator import toy_discriminator
import functions.Basic_tool as bst
import logging, os, sys, time, io
import matplotlib.pyplot as plt 
import numpy as np 
import nibabel as nib 
from nilearn.image import resample_img
import scipy.ndimage 

# set log
logging.basicConfig(filename = 'Execution_record.log', 
                    level = logging.WARNING, 
                    format = '%(filename)s %(message)s')

start_time = time.time()

# Data Dir 
data_dict = {
    "normalised": "./morm",
    "raw": "./raw"
    }

# Set model hyperparameters
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

# Self-defined loss function for GAN
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (_LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss

# tf data mapping function for data augmentation
def tf_random_rotate_image(im, im2):
    def random_rotate_image(image):
        image = scipy.ndimage.rotate(image, np.random.uniform(-60, 60), reshape=False)
        return image
    #im, im2 = images[0], images[1]
    im_shape = im.shape
    im2_shape = im2.shape
    [im,] = tf.py_function(random_rotate_image, [im], [tf.float32])
    im.set_shape(im_shape)
    im2.set_shape(im2_shape)
    return (im,im2)

# Set a callback for the validate data 
class evolRecord(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    predict_img = self.model.generator.predict(ds.take(1).batch(1))
    plt.imshow(predict_img[0, :, :, 64, 0], cmap='gray')
    plt.axis('off')
    plt.savefig(os.path.join(bst.checkpoint_dir, "PrdctImg_Epoch{}.png".format(epoch + 1)), dpi=72)

# Data_preparation using tf data
raw_file_list = [f for f in os.listdir(data_dict["raw"]) if not f.startswith('.')]
raw_file_list.sort()

raw_img_list = []
desired_file_list = []

# Only data with image shape < (128, 128, 128)
for i in range(len(raw_file_list)):
    img = nib.load(os.path.join(data_dict["raw"], raw_file_list[i]))
    img = resample_img(img, target_affine=np.eye(3)*2., interpolation='nearest')
    imag_array = img.get_fdata()
    if imag_array.shape[0] <= 128: 
        imag_array = bst.standardised(imag_array)
        imag_array = imag_array - imag_array.min()
        padded_imag_array = bst.padding_zeros(imag_array, pad_size=128)
        padded_imag_array = padded_imag_array[None,...]
        raw_img_list.append(padded_imag_array)
        desired_file_list.append(raw_file_list[i])
    else:
        pass

# Labels of selected raw data
normed_file_list = [] 
for i in range(len(desired_file_list)):
    img = nib.load(os.path.join(data_dict["normalised"], desired_file_list[i]))
    imag_array = img.get_fdata()
    imag_array = imag_array - imag_array.min()
    padded_imag_array = bst.padding_zeros(imag_array, pad_size=128)
    padded_imag_array = padded_imag_array[None,...]
    normed_file_list.append(padded_imag_array)

raw_img = np.concatenate(raw_img_list, axis = 0)
raw_img = raw_img.astype('float32')
nor_img = np.concatenate(normed_file_list, axis = 0)
nor_img = nor_img.astype('float32')


# Make sure the data fits criteria
if raw_img.shape == nor_img.shape and nor_img.dtype == raw_img.dtype:
    ds = tf.data.Dataset.from_tensor_slices((raw_img, nor_img))
else:
    print("Shape of source img: ", raw_img.shape)
    print("Shape of target img: ", nor_img.shape)
    print("Type of source img: ", raw_img.dtype)
    print("Type of target img: ", nor_img.dtype)
    sys.exit("\033[93m  The size or type of source and target unmatched, check the size again \033[00m")

# Flush out unnecessary memory usage
del imag_array, padded_imag_array, img, raw_file_list, normed_file_list, desired_file_list, raw_img, nor_img


# Set up log file and ckpt location 
os.mkdir(bst.logdir)
os.mkdir(bst.checkpoint_dir)


# Build models
generator = UNet_builder.build_U_Net3D(input_shape, init_fil_Gen, kernel_size)

discriminator = toy_discriminator.build_toy_discriminator(input_shape=dic_input_shape, 
                                                          init_filter_nums=init_fil_Dis, 
                                                          init_kernel_size=kernel_size, 
                                                          kernel_init=kernel_initialiser_dis, 
                                                          repetitions=1)


# Linked the generator and discriminator to create a pix2pix model
pix2pix = GAN(discriminator=discriminator, generator=generator)

# Set up optimisers 
generator_optimizer = tf.keras.optimizers.Adam(g_lr, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(d_lr, beta_1=0.5)


# Set up callbacks
Tensorboard_callbacks = [tf.keras.callbacks.TensorBoard(log_dir=bst.logdir, 
                                                        histogram_freq=1, 
                                                        write_graph=True, 
                                                        write_images=False,
                                                        update_freq='epoch', 
                                                        profile_batch=3, 
                                                        embeddings_freq=0,
                                                        embeddings_metadata=None)]

ImgRecordCallbacks = evolRecord()

# Compile the model 
pix2pix.compile(g_optimizer= generator_optimizer, 
                d_optimizer = discriminator_optimizer,
                loss_fn= [generator_loss, discriminator_loss])

# Fit the model 
hist = pix2pix.fit(ds.skip(10).map(tf_random_rotate_image, num_parallel_calls=32)
                   .shuffle(50).batch(_batchSize), 
                   epochs=_epochs,
                   callbacks=[Tensorboard_callbacks, ImgRecordCallbacks],
                   verbose=1)

# Save the model at the end of training 
pix2pix.save_all(bst.checkpoint_dir, "gen", "disc")
print(f"Training completed log file saved in \033[92m {bst.logdir}\033[00m")
print(f"Training completed model saved in \033[92m {bst.checkpoint_dir}\033[00m")

# Document the records of this trial to log
duration = (time.time() - start_time) /60
logging.warning(f"""Log Path: {bst.logdir}, Ckpt Path: {bst.checkpoint_dir}, Training_Duration: {duration:.2f} mins, 
                    l1_LossLambda: {_LAMBDA}, initail filter number of Generator: {init_fil_Gen}, 
                    initail filter number of Discriminator: {init_fil_Dis}, training in epoch:{_epochs}, 
                    batchSize: {_batchSize}, with G_LR: {g_lr} and D_LR: {d_lr}.""")
