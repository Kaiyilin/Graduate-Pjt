# Resnet 3D model 
from functions.All_functions import *
from functions.cm_tensorboard import *
from functions.Grad_CAM_3D_functions import Grad_CAM_function_2


def importdata_resample(dirr,dirr1,dirr2,dirr3,dirr4,dirr5,pad_size=None):
    def myreadfile_resample_pad(dirr, pad_size):
        #This version can import 3D array regardless of the size
        from nilearn.datasets import load_mni152_template
        from nilearn.image import resample_to_img
        template = load_mni152_template()

        os.chdir(dirr)
        number = 0

        flag = True
        imgs_array = np.array([])
        path_list=[f for f in os.listdir(dirr) if not f.startswith('.')]
        path_list.sort()
        for file in path_list:
            if file.endswith(".nii"):
                #print(os.path.join(dirr, file))
                img = nib.load(os.path.join(dirr, file))
                img_array = resample_to_img(img, template)
                img_array = img.get_fdata()
                #img_array = data_preprocessing(img_array)
                img_array = padding_zeros(img_array, pad_size)
                img_array = img_array.reshape(-1,img_array.shape[0],img_array.shape[1],img_array.shape[2])
                number += 1
                if flag == True:
                    imgs_array = img_array

                else:
                    imgs_array = np.concatenate((imgs_array, img_array), axis=0)

                flag = False
        return number, imgs_array, path_list
    if pad_size == None:
      _, first_mo,  = myreadfile(dirr)
      _, second_mo, _ = myreadfile(dirr1)
      _, third_mo, _ = myreadfile(dirr2)
      
      _, first_mo2, _ = myreadfile(dirr3)
      _, second_mo2, _ = myreadfile(dirr4)
      _, third_mo2, _ = myreadfile(dirr5)
      print(first_mo.shape, second_mo.shape, third_mo.shape, first_mo2.shape, second_mo2.shape, third_mo2.shape)
      return first_mo, second_mo, third_mo, first_mo2, second_mo2, third_mo2

    else:
      _, first_mo, _ = myreadfile_resample_pad(dirr,pad_size)
      _, second_mo, _ = myreadfile_resample_pad(dirr1,pad_size)
      _, third_mo, _ = myreadfile_resample_pad(dirr2,pad_size)
      
      _, first_mo2, _ = myreadfile_resample_pad(dirr3,pad_size)
      _, second_mo2, _ = myreadfile_resample_pad(dirr4,pad_size)
      _, third_mo2, _ = myreadfile_resample_pad(dirr5,pad_size)
      print(first_mo.shape, second_mo.shape, third_mo.shape, first_mo2.shape, second_mo2.shape, third_mo2.shape)
      return first_mo, second_mo, third_mo, first_mo2, second_mo2, third_mo2


def log_confusion_matrix(epoch, logs):
  # Use the model to predict the values from the validation dataset.
  val_pred_raw = model.predict(val_images)
  val_pred = np.argmax(val_pred_raw, axis=1)

  # Calculate the confusion matrix.
  cm = sklearn.metrics.confusion_matrix(val_labels, val_pred)
  # Log the confusion matrix as an image summary.
  figure = plot_confusion_matrix(cm, class_names=class_names)
  cm_image = plot_to_image(figure)

  # Log the confusion matrix as an image summary.
  with file_writer_cm.as_default():
    tf.summary.image("Confusion Matrix", cm_image, step=epoch)



#class_names must defined for cm_tensorboard
class_names = ['HC','C-','C+']

BA_alff, BB_alff, HC_alff, BA_reho, BB_reho, HC_reho = importdata_resample(dir['BA'], dir['BB'], dir['HC'], dir['BA2'], dir['BB2'], dir['HC2'],128)


BA_alff_tr, BA_alff_val = split(5,BA_alff)
BB_alff_tr, BB_alff_val = split(5,BB_alff)
HC_alff_tr, HC_alff_val = split(5,HC_alff)
del BA_alff, BB_alff, HC_alff

BA_reho_tr, BA_reho_val = split(5,BA_reho)
BB_reho_tr, BB_reho_val = split(5,BB_reho)
HC_reho_tr, HC_reho_val = split(5,HC_reho)
del BA_reho, BB_reho, HC_reho

BA_alff_tr, BA_alff_val = BA_alff_tr[...,None], BA_alff_val[...,None]
BB_alff_tr, BB_alff_val = BB_alff_tr[...,None], BB_alff_val[...,None]
HC_alff_tr, HC_alff_val = HC_alff_tr[...,None], HC_alff_val[...,None]

BA_reho_tr, BA_reho_val = BA_reho_tr[...,None], BA_reho_val[...,None]
BB_reho_tr, BB_reho_val = BB_reho_tr[...,None], BB_reho_val[...,None]
HC_reho_tr, HC_reho_val = HC_reho_tr[...,None], HC_reho_val[...,None]

BA_labels_tr = np.ones(BA_alff_tr.shape[0])
BA_labels_val = np.ones(BA_alff_val.shape[0])

HC_labels_tr = np.zeros(HC_alff_tr.shape[0])
HC_labels_val = np.zeros(HC_alff_val.shape[0])

def tfdata_shuffle_and_split(images, labels, split_number):
    images, labels = shuffle(images, labels)
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    ds_tr = ds.skip(split_number)
    ds_val = ds.take(split_number)
    return ds_tr, ds_val

# tr image channel concatenate
BA_ds_tr, BA_ds_val = tfdata_shuffle_and_split(BA_alff_tr, BA_labels_tr, 5)
HC_ds_tr, HC_ds_val = tfdata_shuffle_and_split(HC_alff_tr, HC_labels_tr, 5)

# should I using prefetch?
ds_tr = tf.data.Dataset.concatenate(BA_ds_tr,HC_ds_tr)
ds_val = tf.data.Dataset.concatenate(BA_ds_val,HC_ds_val)


# val image channel concatenate
val_images = np.concatenate([BA_alff_val, HC_alff_val],axis=0)
val_labels = np.concatenate([BA_labels_val,HC_labels_val],axis=0)

if len(val_labels) == len(val_images) and val_images.ndim == 5:
    print("\nsample size of val_data is equivalent.")
else:
    sys.exit('\ncheck the size of your val data')

del BA_alff_tr, BA_alff_val, BA_reho_tr,BA_reho_val
del BB_alff_tr, BB_alff_val, BB_reho_tr,BB_reho_val
del HC_alff_tr, HC_alff_val, HC_reho_tr,HC_reho_val

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
            image = scipy.ndimage.rotate(image, np.random.uniform(-90, 90), axes=(0,1), reshape=False)
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
        image = scipy.ndimage.rotate(image, np.random.uniform(-60, 60), reshape=False)
        return image
    im_shape = image.shape
    [image,] = tf.py_function(random_rotate_image, [image], [tf.float64])
    image.set_shape(im_shape)
    return image, label


_buffer_size = 120
_batch_size = 5


from resnet3d import Resnet3DBuilder 
opt3 = tf.keras.optimizers.SGD(lr=1e-3, momentum=0.9, nesterov=True)
opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-3 / 200)
project_name =  input('Naming this project: ')

strategy = tf.distribute.MirroredStrategy()
print('\nNumber of GPU devices: {}'.format(strategy.num_replicas_in_sync))
os.mkdir(logdir)
logdir = logdir + project_name
os.mkdir(logdir)
os.mkdir(checkpoint_dir)
# Define the per-epoch callback.
file_writer_cm = tf.summary.create_file_writer(logdir + 'cm')
cm_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)

with strategy.scope():
    os.chdir(logdir)
    model = Resnet3DBuilder.build_resnet_101((128, 128, 128, 1), 2,reg_factor=1e-4)
    print("\nBuilding a Res_Net50 model")
    model.compile(optimizer= opt3 , loss=loss, metrics=['sparse_categorical_accuracy'])
    print(model.summary())

    class PrintLR(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print('\nLearning rate for epoch {} is {}'.format(epoch + 1,
                                                      model.optimizer.lr.numpy()))
                
    callbacks = [ tf.keras.callbacks.TensorBoard(log_dir=logdir, 
                                                    histogram_freq=1, 
                                                    write_graph=True, 
                                                    write_images=False,
                                                    update_freq='epoch', 
                                                    profile_batch=2, 
                                                    embeddings_freq=0,
                                                    embeddings_metadata=None),
                tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                                   verbose=0,
                                                   save_weights_only=True,
                                                   save_freq='epoch'),
                cm_callback,
                PrintLR()]

    history = model.fit( ds_tr.shuffle(buffer_size=_buffer_size).map(tf_random_rotate_image, num_parallel_calls=32).batch(_batch_size).prefetch(_batch_size),
                             class_weight = None,
                             epochs=250,
                             verbose=1,
                             callbacks=callbacks,
                             validation_split=None,
                             validation_data= ds_val.batch(_batch_size))

#Grad_CAM_function_2(val_images,val_labels,model,Grad_CAM_save_path)