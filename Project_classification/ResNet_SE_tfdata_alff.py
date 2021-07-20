 # Resnet 3D model 
from functions.All_functions import *
from functions.cm_tensorboard import *
from Data_alff_BAHC_64_preparation import *
import logging

logging.basicConfig(filename = '/home/kaiyi/Project_classification/Execution_record.log', level = logging.WARNING, format = '%(filename)s:%(message)s')

# Data Preparation

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

class_names = ['HC', 'C+']


_buffer_size = 120
_batch_size = 5
_lr = 1e-02
_num_classes = 2

from resnet3d_variation.resnet3dSE_swish import ResnetSEswish_3DBuilder 
loss = tf.keras.losses.BinaryCrossentropy()
#opt = tf.keras.optimizers.Adam(lr=1e-3)
opt3 = tf.keras.optimizers.SGD(lr= _lr, momentum=0.9, nesterov=True)

# Setting Strategy
strategy = tf.distribute.MirroredStrategy()
print('\nNumber of GPU devices: {}'.format(strategy.num_replicas_in_sync))

# Creating file_saving path
os.mkdir(logdir)
logdir = f"{logdir}/ResSwishSE_lr_{_lr}_num_classes{_num_classes}"
os.mkdir(logdir)
os.mkdir(checkpoint_dir)


# Define the per-epoch callback.
file_writer_cm = tf.summary.create_file_writer(logdir + 'cm')
cm_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)


with strategy.scope():
    model = ResnetSEswish_3DBuilder.build_resnet_50((64, 64, 64, 1), _num_classes, reg_factor=1e-4)
    print("Building a Res_Net_50 model")
    
    model.compile(optimizer = opt3 , loss = loss, metrics = [tf.keras.metrics.BinaryAccuracy(name='binary_accuracy', dtype=None, threshold=0.5),tf.keras.metrics.AUC()])
    print(model.summary())

            

            
    callbacks = [tf.keras.callbacks.TensorBoard(log_dir = logdir, 
                                                histogram_freq = 1, 
                                                write_graph = True, 
                                                write_images = False,
                                                update_freq = 'epoch', 
                                                profile_batch = 2, 
                                                embeddings_freq = 0,
                                                embeddings_metadata = None),
                 tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_prefix,
                                                    verbose = 0,
                                                    save_weights_only = True,
                                                    save_freq = 'epoch'),
                cm_callback]

    history = model.fit(ds_tr.shuffle(buffer_size=_buffer_size).map(tf_random_rotate_image, num_parallel_calls=32).batch(_batch_size).prefetch(_batch_size),
                        class_weight = None,
                        epochs = 200,
                        verbose = 1,
                        callbacks = callbacks,
                        validation_split = None,
                        validation_data = ds_val.batch(11))

logging.warning(
                f"""
                ResSwishSE_lr_{_lr}_num_classes{_num_classes}, logdir: {logdir}, ckptdir: {checkpoint_dir}, 
                """)