#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 13:58:47 2019

@author: yixiaowan

To get VGG16 baseline performance for the chosen classes.

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

num_epochs = 100
bs = 64
img_rows = 224
img_cols = 224
classes_list = ['ave', 'canidae', 'cloth', 'felidae', 'kitchen', 'land_trans']



imagenet_test = '/mnt/fast-data16/datasets/ILSVRC/2012/clsloc/val/'
model = VGG16(weights = 'imagenet', include_top=True, input_shape = (img_rows, img_cols, 3))
baseline_df = pd.DataFrame(columns = ['class_name', 'ic_acc_baseline', 'oc_acc_baseline'])
for i in range(len(classes_list)):
    class_name = classes_list[i]
    class_csv_path = 'groupings-csv/' + class_name + '_Imagenet.csv'
    df_classes = pd.read_csv(class_csv_path, usecols=['wnid'])

    classes = sorted([i for i in df_classes['wnid']])
    whole_list = os.listdir(imagenet_test)
    oc_classes = sorted([i for i in whole_list if i not in classes])


    ImageGen = ImageDataGenerator(fill_mode='nearest',
                              horizontal_flip=False,
                              rescale=None,
                              preprocessing_function=preprocess_input,
                              data_format="channels_last",
                              )


    in_context_generator, in_context_steps = create_good_generator(ImageGen,
                                                    imagenet_test,
                                                    batch_size=bs,
                                                    target_size = (img_rows, img_cols),
                                                    class_mode='sparse',
                                                    AlextNetAug=False, 
                                                    classes=classes)


    out_context_generator, out_context_steps = create_good_generator(ImageGen,
                                                    imagenet_test,
                                                    batch_size=bs,
                                                    target_size = (img_rows, img_cols),
                                                    class_mode='sparse',
                                                    AlextNetAug=False, 
                                                    classes=oc_classes)



    ic_loss, ic_acc = model.evaluate_generator(in_context_generator, in_context_steps, verbose=1)
    oc_loss, oc_acc = model.evaluate_generator(out_context_generator, out_context_steps, verbose=1)
    

save_path = 'single_att_results/' + 'baseline.csv'
baseline_df.to_csv(save_path)    
    
    
    
    
    
    
    
    
    
    
    