# mnist test

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
import datetime
from tensorflow.keras import models
from tensorflow.keras import layers
import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from All_functions import *
import io
import itertools
from six.moves import range
from cm_tensorboard import *

class_names = ['0','1','2','3','4','5','6','7','8','9']

def log_confusion_matrix(epoch, logs):
  # Use the model to predict the values from the validation dataset.
  test_pred_raw = model.predict(test_images)
  test_pred = np.argmax(test_pred_raw, axis=1)

  # Calculate the confusion matrix.
  cm = sklearn.metrics.confusion_matrix(test_labels, test_pred)
  # Log the confusion matrix as an image summary.
  figure = plot_confusion_matrix(cm, class_names=class_names)
  cm_image = plot_to_image(figure)

  # Log the confusion matrix as an image summary.
  with file_writer_cm.as_default():
    tf.summary.image("Confusion Matrix", cm_image, step=epoch)


# tensorboard log directiory
logdir="/Users/kaiyi/Desktop/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Define the per-epoch callback.
file_writer_cm = tf.summary.create_file_writer(logdir + '/cm')
cm_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)

# model checkpoint
checkpoint_dir = '/Users/kaiyi/Desktop/'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_prefix = os.path.join(checkpoint_dir, "weights.{epoch:02d}.hdf5")

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


def get_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10,activation='softmax'))
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = get_model()

train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255
train_images = train_images[...,None]
test_images = test_images[...,None]


train_datasets = tf.data.Dataset.from_tensor_slices((train_images,train_labels))
train_datasets = train_datasets.shuffle(buffer_size=len(train_images)).batch(300)

test_datasets = tf.data.Dataset.from_tensor_slices((test_images,test_labels))
test_datasets = test_datasets.shuffle(buffer_size=len(test_images)).batch(300)

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


hist= model.fit(train_datasets,  
                 epochs=3,
                 callbacks = callbacks,
                 validation_split = None,
                 validation_data = test_datasets)


#another = network.predict(test_images,verbose=1)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

x = test_images[1]
x_2 = x[None,...] 

predictions = model.predict(x_2)

output = model.output[:, 2]

#last_conv_layer = model.get_layer('conv2d_2') #Gradient of the desire class with regard to the output feature map of conv2d_2
grad_model = tf.keras.models.Model([model.inputs], [model.get_layer('conv2d_2').output, model.output])

with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(x_2)
    loss = predictions[:, 2]  

output = conv_outputs[0]
grads = tape.gradient(loss, conv_outputs)[0]


# Average gradients spatially
weights = tf.reduce_mean(grads, axis=(0,1))
# Build a ponderated map of filters according to gradients importance
cam = np.zeros(output.shape[0:2], dtype=np.float32)

for index, w in enumerate(weights):
    cam += w * output[:, :, index]


from skimage.transform import resize
from matplotlib import pyplot as plt
capi=resize(cam,(28,28))
print(capi.shape)
capi = np.maximum(capi,0)
heatmap = (capi - capi.min()) / (capi.max() - capi.min())

y = x[:,:,0]

f, axarr = plt.subplots(1,3,figsize=(15,10))
f.suptitle('Grad-CAM')

img_plot = axarr[0].imshow(y, cmap='gray');
axarr[0].axis('off')
axarr[0].set_title('Numbers')

img_plot = axarr[1].imshow(heatmap, cmap='jet');
axarr[1].axis('off')
axarr[1].set_title('Heatmap')

overlay=cv2.addWeighted(y,0.3,heatmap, 0.6, 0, dtype = cv2.CV_32F)

img_plot = axarr[2].imshow(overlay, cmap='jet');
axarr[2].axis('off')
axarr[2].set_title('Overlap')