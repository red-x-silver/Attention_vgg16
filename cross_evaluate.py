#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 01:47:18 2019

@author: yixiaowan

To evaluate different category-attended model's performance over other categories.

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
import json

#Forked from Ken's
import keras_custom_objects as KO


bs = 64
img_rows = 224
img_cols = 224
imagenet_test = '/mnt/fast-data16/datasets/ILSVRC/2012/clsloc/val/'

ImageGen = ImageDataGenerator(fill_mode='nearest',
                              horizontal_flip=False,
                              rescale=None,
                              preprocessing_function=preprocess_input,
                              data_format="channels_last",
                              )



ctgry_list = ['ave', 'canidae', 'cloth', 'felidae', 'kitchen', 'land_trans']
performance_dict = {}


for crgry in ctgry_list:
    model_file = ctgry + '_models/'
    model_path_list = os.listdir(model_file)
    performance = []
    for i in range(len(model_path_list)):
        model_path = model_path_list[i]
        model = load_model(model_file+model_path, custom_objects={'SinglyConnected': SinglyConnected, 'CustomModel': KO.CustomModel})

        for class_name in ctgry_list:
            if class_name != ctgry:
                
                
                class_csv_path = 'groupings-csv/' + class_name + '_Imagenet.csv'
                df_classes = pd.read_csv(class_csv_path, usecols=['wnid'])
                classes = sorted([i for i in df_classes['wnid']])

                in_context_generator, in_context_steps = create_good_generator(ImageGen,
                                                                           imagenet_test,
                                                                           batch_size=bs,
                                                                           target_size = (img_rows, img_cols),
                                                                           class_mode='sparse',
                                                                           AlextNetAug=False, 
                                                                           classes=classes)
                ic_loss, ic_acc = model.evaluate_generator(in_context_generator, in_context_steps, verbose=1)
                performance.append([model_path[:-3], class_name, ic_acc])
                
    
    performance_dict[ctgry] = performance
    
print (performance_dict)

with open('cross_evaluation.json', 'w') as fp:
    json.dump(performance_dict, fp)
            

