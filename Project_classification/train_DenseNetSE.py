import pymongo, logging
from functions.All_functions import *
from Data_preparation import *
from functions.cm_tensorboard import *
from functions.classVis import IntegratedGradients
from densenet3d.DenseNet_swish_se import DenseNet3Dbuilder 
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score, accuracy_score, f1_score


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


logging.basicConfig(filename = '/home/kaiyi/Project_classification/Execution_record.log', level = logging.WARNING, format = '%(filename)s:%(message)s')

class_names = ['HC', 'C+']
_buffer_size = 120
_batch_size = 5
_num_classes = 2
_bottleneckRatio = 4
_gr = 16
_lr = 0.01


loss = tf.keras.losses.BinaryCrossentropy()
opt3 = tf.keras.optimizers.SGD(lr=_lr, momentum=0.9, nesterov=True)

# Setting Strategy
strategy = tf.distribute.MirroredStrategy()
print('\nNumber of GPU devices: {}'.format(strategy.num_replicas_in_sync))

# Creating ffile_saving path
os.mkdir(logdir)
logdir = f"{logdir}/Dense_gr{_gr}_bottleneckRatio{_bottleneckRatio}_lr_{_lr}_num_classes{_num_classes}"
os.mkdir(logdir)
os.mkdir(checkpoint_dir)

# Define the per-epoch callback.
file_writer_cm = tf.summary.create_file_writer(logdir + 'cm')
cm_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)


with strategy.scope():
    model = DenseNet3Dbuilder.build_densenet_121(input_shape = (64, 64, 64, 1), n_classes= 2, bottleneck_ratio= _bottleneckRatio , growth_rate= _gr)
    print("Building a DenseNet model")
    model.compile(optimizer= opt3 , loss=loss, metrics=[tf.keras.metrics.BinaryAccuracy(name='binary_accuracy', dtype=None, threshold=0.5), tf.keras.metrics.AUC()])
    print(model.summary())

            

            
    callbacks = [tf.keras.callbacks.TensorBoard(log_dir=logdir, 
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
                Dense121SwishSE_lr_{_lr}_num_classes{_num_classes}, logdir: {logdir}, ckptdir: {checkpoint_dir}, growth_rate {_gr}, bottleNeckratio: {_bottleneckRatio}.
                """)
