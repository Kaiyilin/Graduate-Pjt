# mnist test

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras.datasets import mnist
import os
import numpy as np
#import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
import io
import itertools
#from six.moves import range

#os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

def get_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding = 'same', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu',padding = 'same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu',padding = 'same'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding = 'same'))
    model.add(Conv2D(512, (3, 3), activation='relu',padding = 'same'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(10,activation='softmax'))
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


model = get_model()

train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255
train_images = train_images[...,None]
test_images = test_images[...,None]


train_datasets = tf.data.Dataset.from_tensor_slices((train_images,train_labels))
train_datasets = train_datasets.shuffle(buffer_size=len(train_images)).batch(300).prefetch(3000)

test_datasets = tf.data.Dataset.from_tensor_slices((test_images,test_labels))
test_datasets = test_datasets.shuffle(buffer_size=len(test_images)).batch(300)


strategy = tf.distribute.MirroredStrategy()
print('\nNumber of GPU devices: {}'.format(strategy.num_replicas_in_sync))


with strategy.scope():
    model = get_model()


    train_datasets = tf.data.Dataset.from_tensor_slices((train_images,train_labels))

    test_datasets = tf.data.Dataset.from_tensor_slices((test_images,test_labels))
    test_datasets = test_datasets.shuffle(buffer_size=len(test_images)).batch(1000)


    hist= model.fit(train_datasets.shuffle(buffer_size=len(train_images)).skip(10000).batch(1000).prefetch(8000),  
                    epochs=5,
                    callbacks = None,
                    validation_split = None,
                    validation_data = train_datasets.take(10000).batch(100))

df = pd.DataFrame(hist.history)
df.to_csv("relu.csv")

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc:', test_acc)