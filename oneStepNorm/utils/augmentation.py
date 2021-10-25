import scipy
import numpy as np
import tensorflow as tf

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