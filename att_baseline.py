#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 23:53:47 2019

Baseline model of attention layer inserted at different positions.

First attempt is to train with position layer06.

Training example: python att_baseline --position 06

@author: yixiaowan
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import matplotlib
matplotlib.use("Agg")

import keras
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Dense, Flatten, Reshape, Concatenate, Lambda
from keras import optimizers
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras import backend as K
from keras.layers import Layer

import numpy as np
import pandas as pd
import h5py

import random
import math


from custom_layer_constraints import CustomConstraint, SinglyConnected

#Forked from Ken's
import keras_custom_objects as KO
from custom_generator import create_good_generator


import argparse
parser = argparse.ArgumentParser(
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--position', type=str, default='06')
args = parser.parse_args()
position = args.position
print (position)
assert (position in ['03', '06', '10', '14', '18', '20', '21']), "only support positions from ['03', '06', '10', '14', '18', '20', '21']"

#Magic numbers
num_epochs = 100
bs = 64
img_rows = 224
img_cols = 224
patience = 2

imagenet_train = '/mnt/fast-data16/datasets/ILSVRC/2012/clsloc/train/'
ImageGen = ImageDataGenerator(fill_mode='nearest',
                              horizontal_flip=True,
                              rescale=None,
                              preprocessing_function=preprocess_input,
                              data_format="channels_last",
                              validation_split=0.1
                              )
good_train_generator, steps = create_good_generator(ImageGen, 
                                                    imagenet_train,
                                                    batch_size=bs, 
                                                    target_size = (img_rows, img_cols), 
                                                    class_mode='sparse', 
                                                    subset= 'training', 
                                                    )

good_validation_generator, steps_val = create_good_generator(ImageGen, 
                                                    imagenet_train,
                                                    batch_size=bs, 
                                                    target_size = (img_rows, img_cols), 
                                                    class_mode='sparse', 
                                                    subset= 'validation', 
                                                    )

#Build the model
layer_index = int(position)

frozen_model = VGG16(weights = 'imagenet', include_top=True, input_shape = (img_rows, img_cols, 3))
for layer in frozen_model.layers:
    layer.trainable = False
    
    
last = frozen_model.layers[layer_index].output

#By default, weights in SinglyConnected are initialized to be ones.
x = SinglyConnected(kernel_constraint= CustomConstraint())(last)
  
for layer in frozen_model.layers[layer_index+1:]:
    x = layer(x)    
    
model = KO.CustomModel(frozen_model.input, x)
  
model.summary()
  
model.compile(loss='sparse_categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

#Callbacks  
custom_earlystopping = KO.RelativeEarlyStopping(monitor='val_loss',
                                                  min_perc_delta=0.001,  # perc means percentage
                                                  patience=patience,
                                                  verbose=1,
                                                  mode='min'
                                                  )
file_name = 'baseline_models/baseline_model_layer'+position + '.h5'                                                 
log_path = './baseline_logs'
tensorboard = TensorBoard(log_dir = log_path, update_freq = bs*1000)
callbacks = [tensorboard,
             custom_earlystopping,
             ModelCheckpoint(filepath=file_name, monitor='val_loss', save_best_only=True)]

  
#Training    
model.fit_generator_custom(
          good_train_generator,
          steps_per_epoch=steps,
          epochs=num_epochs,
          verbose = 1, 
          callbacks = callbacks, 
          validation_data=good_validation_generator, 
          validation_steps=steps_val,
          #max_queue_size=20,
          #workers = 6,
          use_multiprocessing=False
          )


