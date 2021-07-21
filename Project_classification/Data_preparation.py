# Data preparation

from functions.All_functions import *

def split_and_channel(c,array):
    array_val = array[:c,:,:,:]
    array_tr = array[c:,:,:,:]

    array_tr = array_tr[...,None]
    array_val = array_val[...,None]
    return array_tr, array_val
class_names = ['HC','C+']

BA_alff, _, HC_alff, _, _, _ = importdata2(dir['BA'],dir['BB'],dir['HC'],dir['BA2'],dir['BB2'],dir['HC2'],64)

BA_alff_tr, BA_alff_val = split_and_channel(5,BA_alff)
HC_alff_tr, HC_alff_val = split_and_channel(5,HC_alff)
del BA_alff, HC_alff

BA_labels_tr = np.ones(BA_alff_tr.shape[0])
BA_labels_val = np.ones(BA_alff_val.shape[0])

HC_labels_tr = np.zeros(HC_alff_tr.shape[0])
HC_labels_val = np.zeros(HC_alff_val.shape[0])

def tfdata_shuffle_and_split(images, labels, split_number, num_class):
    images, labels = shuffle(images, labels)
    labels = tf.keras.utils.to_categorical(labels, num_classes = num_class)
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    ds_tr = ds.skip(split_number)
    ds_val = ds.take(split_number)
    return ds_tr, ds_val

BA_ds_tr, BA_ds_val = tfdata_shuffle_and_split(BA_alff_tr, BA_labels_tr, 5, 2)
HC_ds_tr, HC_ds_val = tfdata_shuffle_and_split(HC_alff_tr, HC_labels_tr, 6, 2)

ds_tr = tf.data.Dataset.concatenate(BA_ds_tr, HC_ds_tr)
ds_val = tf.data.Dataset.concatenate(BA_ds_tr, HC_ds_val)

# Disable AutoShard.
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
ds_tr = ds_tr.with_options(options)
ds_val = ds_val.with_options(options)

val_images = np.concatenate([BA_alff_val, HC_alff_val],axis=0)
val_labels = np.concatenate([BA_labels_val, HC_labels_val],axis=0)

if len(val_labels) == len(val_images) and val_images.ndim == 5:
    print("\nsample size of val_data is equivalent.")
else:
    sys.exit('\ncheck the size of your val data')

def tf_random_rotate_image_xyz(image, label):
    # 3 axes random rotation
    def rotateit_y(image):
        toggleSwitch = bool(random.getrandbits(1))

        if toggleSwitch == True:
            image = scipy.ndimage.rotate(image, np.random.uniform(-60, 60), axes=(0,2), reshape=False)
        else:
            image = image
        return image

    def rotateit_x(image):
        toggleSwitch = bool(random.getrandbits(1))

        if toggleSwitch == True:
            image = scipy.ndimage.rotate(image, np.random.uniform(-60, 60), axes=(1,2), reshape=False)     
        else:
            image = image  
        return image

    def rotateit_z(image):
        toggleSwitch = bool(random.getrandbits(1))
        
        if toggleSwitch == True:
            image = scipy.ndimage.rotate(image, np.random.uniform(-60, 60), axes=(0,1), reshape=False)
        else:
            image = image
        return image

    im_shape = image.shape
    [image,] = tf.py_function(rotateit_x, [image], [tf.float64])
    [image,] = tf.py_function(rotateit_y, [image], [tf.float64])
    [image,] = tf.py_function(rotateit_z, [image], [tf.float64])
    image.set_shape(im_shape)
    return image, label

# one axes random rotation
def tf_random_rotate_image(image, label):
    def random_rotate_image(image):
        image = scipy.ndimage.rotate(image, np.random.uniform(-80, 80), order=0, reshape=False)
        return image
    im_shape = image.shape
    [image,] = tf.py_function(random_rotate_image, [image], [tf.float64])
    image.set_shape(im_shape)
    return image, label