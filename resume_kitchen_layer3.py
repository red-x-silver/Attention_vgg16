# -*- coding: utf-8 -*-
"""all_layers_kitchen.ipynb
    Automatically generated by Colaboratory.
    Original file is located at
    https://colab.research.google.com/drive/1Ah1FAqm098v6oR6mRgYDnItur8L532Jw
    """

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import matplotlib
matplotlib.use("Agg")

import keras
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Dense, Flatten, LocallyConnected1D, LocallyConnected2D, Reshape, Concatenate, Lambda
from keras import optimizers
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.layers import Layer

import numpy as np
import pandas as pd
import h5py

import random
import math

from custom_generator import create_good_generator
from keras.models import load_model
from custom_layer_constraints import CustomConstraint, SinglyConnected

#Magic numbers
num_epochs = 100
bs = 64
img_rows = 224
img_cols = 224
flatten_shape = 1000

# Forked from Ken's codes

imagenet_train = '/mnt/fast-data16/datasets/ILSVRC/2012/clsloc/train/'
ImageGen = ImageDataGenerator(fill_mode='nearest',
                              horizontal_flip=True,
                              rescale=None,
                              preprocessing_function=preprocess_input,
                              data_format="channels_last",
                              validation_split=0.1
                              )
df_classes = pd.read_csv('groupings-csv/kitchen_Imagenet.csv', usecols=['wnid'])
classes = sorted([i for i in df_classes['wnid']])

good_train_generator, steps = create_good_generator(ImageGen,
                                                    imagenet_train,
                                                    batch_size=bs,
                                                    target_size = (img_rows, img_cols),
                                                    class_mode='sparse',
                                                    subset= 'training',
                                                    classes=classes)

good_validation_generator, steps_val = create_good_generator(ImageGen,
                                                             imagenet_train,
                                                             batch_size=bs,
                                                             target_size = (img_rows, img_cols),
                                                             class_mode='sparse',
                                                             subset= 'validation',
                                                             classes=classes)

file_name = 'layer' + str(3) + '-kitchen-model-CustomLayer.h5'
model = load_model('kitchen/'+file_name, custom_objects={'SinglyConnected': SinglyConnected,
                                                        'CustomConstraint': CustomConstraint})


print('This model is going to be saved in: ' + file_name)
                  
                  
                  
#Callbacks
callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose = 1),
            ModelCheckpoint(filepath='kitchen/'+file_name, monitor='val_loss', save_best_only=True)]
                      
                      
                      
print('Now training model with layer index: '+ str(layer_index))
                      
model.fit_generator(
                    good_train_generator,
                    steps_per_epoch=steps,
                    epochs=num_epochs,
                    verbose = 1,
                    callbacks = callbacks,
                    validation_data=good_validation_generator,
                    validation_steps=steps_val
                    )
