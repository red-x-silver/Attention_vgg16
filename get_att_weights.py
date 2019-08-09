#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 10:02:20 2019

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
from keras.layers import Dense, Flatten, LocallyConnected1D, LocallyConnected2D, Reshape, Concatenate, Lambda
from keras import optimizers
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.layers import Layer

from keras.models import load_model
from custom_generator import create_good_generator
from custom_layer_constraints import CustomConstraint, SinglyConnected
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Forked from Ken's
import keras_custom_objects as KO


def get_att_weights(model_file):
  model_path_list = os.listdir(model_file)
  model_path_list = [x for x in model_path_list if x[-4]=='l'] #filter out models with names as 'xxx-CustomLayer.h5', only use models with 'xxx-model.h5'
  for i in range(len(model_path_list)):
    model_path = model_path_list[i]
    model = load_model(model_file+model_path, custom_objects={'SinglyConnected': SinglyConnected, 'CustomModel': KO.CustomModel})
    att_weights = np.array(model.get_layer('singly_connected_1').get_weights())
    save_path = 'att_wights/' + model_path[:-9]+'weights.npy'
    print ('weights saves as' + save_path)
    np.save(save_path,att_weights)
  return None

for class_name in ['ave', 'canidae', 'cloth', 'felidae', 'kitchen', 'land_trans']:
    model_file = class_name + '_models/'
    get_att_weights(model_file)