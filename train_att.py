# -*- coding: utf-8 -*-
"""
Script to train Vgg_16 with one single attention layer inserted at different positions.

Specify class_name from [ave, canidae, cloth, felidae, kitchen, land_trans] when run the script.

Make sure you have the class_name+'_models' directory before training.

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
parser.add_argument('--class_name', type=str, default='kitchen')
print (class_name)
assert (class_name in ['ave', 'canidae', 'cloth', 'felidae', 'kitchen', 'land_trans']), "only support class from ['ave', 'canidae', 'cloth', 'felidae', 'kitchen', 'land_trans']"

#Magic numbers
num_epochs = 100
bs = 64
img_rows = 224
img_cols = 224
layer_indice = ['21', '03', '06', '10', '14', '18', '20']  #positions to insert attention layer,use dtr here for writing file name properly(layer03 rather than layer3)

# Custom generator from Ken's codes
imagenet_train = '/mnt/fast-data16/datasets/ILSVRC/2012/clsloc/train/'
ImageGen = ImageDataGenerator(fill_mode='nearest',
                              horizontal_flip=True,
                              rescale=None,
                              preprocessing_function=preprocess_input,
                              data_format="channels_last",
                              validation_split=0.1
                              )

class_csv_path = 'groupings-csv/' + class_name + '_Imagenet.csv'
df_classes = pd.read_csv(class_csv_path, usecols=['wnid'])
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

#Build the model

def get_ATT_model(str_layer_index, class_name):
  '''
  The function is to insert an attention layer 
  between frozen_model.layers[layer_index] and frozen_model.layers[layer_index+1].
  frozen_model is set to be VGG16.
  
  Input str_layer_index is a string. This is to have proper model_file_name, e.g. layer03 rather than layer3
  '''
  layer_index = int(str_layer_index)
  frozen_model = VGG16(weights = 'imagenet', include_top=True, input_shape = (img_rows, img_cols, 3))
  for layer in frozen_model.layers:
    layer.trainable = False
    
    
  last = frozen_model.layers[layer_index].output

  x = SinglyConnected(kernel_constraint= CustomConstraint())(last)
  
  for layer in frozen_model.layers[layer_index+1:]:
    x = layer(x)    
    
  model = KO.CustomModel(frozen_model.input, x)
  
  model.summary()
  
  model_file_name = class_name+ '_models/layer' + str_layer_index + '-' + class_name + '-model-CustomLayer.h5'

  return model, model_file_name



for str_layer_index in layer_indice:
  
  model, file_name = get_ATT_model(str_layer_index, class_name)
  model.compile(loss='sparse_categorical_crossentropy', 
                optimizer='adam', 
                metrics=['accuracy'])
  
  print('This model is going to be saved in: ' + file_name)
  
  
  
  #Callbacks
  custom_earlystopping = KO.RelativeEarlyStopping(monitor='val_loss',
                                                  min_perc_delta=0.001,  # perc means percentage
                                                  patience=patience,
                                                  verbose=2,
                                                  mode='min'
                                                  )
                                                  
  log_path = './'+class_name+'_logs'
  tensorboard = TensorBoard(log_dir = log_path, update_freq = bs*100)
  callbacks = [tensorboard,
               custom_earlystopping,
               ModelCheckpoint(filepath=file_name, monitor='val_loss', save_best_only=True)]

  
  
  print('Now training model with layer index: '+ str_layer_index)
  
  model.fit_generator_custom(
          good_train_generator,
          steps_per_epoch=steps,
          epochs=num_epochs,
          verbose = 1, 
          callbacks = callbacks, 
          validation_data=good_validation_generator, 
          validation_steps=steps_val,
          max_queue_size=40,
          workers=14,
          use_multiprocessing=True
          )
