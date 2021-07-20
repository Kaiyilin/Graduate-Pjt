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
#os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['accuracy'])

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

from tensorflow.keras.utils import to_categorical 

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

train_datasets = tf.data.Dataset.from_tensor_slices((train_images,train_labels))
train_datasets = train_datasets.shuffle(buffer_size=len(train_images)).batch(300)

#logdir = "/Users/kaiyi/Dropbox/Learning/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#logdir="/home/user/Desktop/mult_channel/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist= network.fit(train_datasets,  
                  epochs=5,
                  callbacks = None,
                  validation_split = None)


#another = network.predict(test_images,verbose=1)

test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)
