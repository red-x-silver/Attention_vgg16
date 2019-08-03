# -*- coding: utf-8 -*-
"""
Evaluate trained models based on in-context, out-context, whole accuracies.

python evaluate_models --class_name kitchen
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

#Forked from Ken's
import keras_custom_objects as KO


import argparse
parser = argparse.ArgumentParser(
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--class_name', type=str, default='kitchen')
args = parser.parse_args()
class_name = args.class_name
assert class_name in ['ave', 'canidae', 'cloth', 'felidae', 'kitchen', 'land_trans'], "only support class from ['ave', 'canidae', 'cloth', 'felidae', 'kitchen', 'land_trans']"

bs = 64
img_rows = 224
img_cols = 224
imagenet_test = '/mnt/fast-data16/datasets/ILSVRC/2012/clsloc/val/'

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


whole_set_gen, whole_set_steps = create_good_generator(ImageGen,
                                                    imagenet_test,
                                                    batch_size=bs,
                                                    target_size = (img_rows, img_cols),
                                                    class_mode='sparse',
                                                    AlextNetAug=False, 
                                                    )

def auto_evaluate(model_file):
  #eval_df = pd.DataFrame(columns = ['model_name', 'ic_loss', 'oc_loss', 'whole_loss', 'ic_acc', 'oc_acc', 'whole_acc'])
  eval_df = pd.DataFrame(columns = ['model_name', 'ic_loss', 'oc_loss', 'ic_acc', 'oc_acc'])
  model_path_list = os.listdir(model_file)
  model_path_list = [x for x in model_path_list if x[-4]=='l'] #filter out models with names as 'xxx-CustomLayer.h5', only use models with 'xxx-model.h5'
  for i in range(len(model_path_list)):
    model_path = model_path_list[i]
    model = load_model(model_file+model_path, custom_objects={'SinglyConnected': SinglyConnected, 'CustomModel': KO.CustomModel})
    ic_loss, ic_acc = model.evaluate_generator(in_context_generator, in_context_steps, verbose=1)
    oc_loss, oc_acc = model.evaluate_generator(out_context_generator, out_context_steps, verbose=1)
    #whole_loss, whole_acc = model.evaluate_generator(whole_set_gen, whole_set_steps, verbose=1)
    eval_df.loc[i] = {'model_name': model_path[:-3], 
                      'ic_loss': ic_loss, 
                      'oc_loss': oc_loss, 
                      #'whole_loss': whole_loss, 
                      'ic_acc': ic_acc, 
                      'oc_acc': oc_acc
                      #'whole_acc': whole_acc
                     }
  eval_df.sort_values(by=['model_name'], inplace=True)
  return eval_df




model_file = class_name + '_models/'
eval_df = auto_evaluate(model_file)
print (eval_df)
save_path = 'results_df/' + class_name + '.csv'
eval_df.to_csv(save_path)
