# -*- coding: utf-8 -*-
"""evaluate_canidae_models.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VLMV9yWAYuelmC6IOliEj7mT6tJeRv34
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

import numpy as np
import pandas as pd

bs = 64
img_rows = 224
img_cols = 224

#Customize a element-wise multiplication layer with trainable weights
class SinglyConnected(Layer):
    def __init__(self,
                 kernel_constraint=None,
                 **kwargs):
        self.kernel_constraint = kernel_constraint
        super(SinglyConnected, self).__init__(**kwargs)

    def build(self, input_shape):
        if input_shape[-1] is None:
            raise ValueError('Axis ' +  + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')
        #self.input_spec = InputSpec(ndim=len(input_shape),
         #                           axes=dict(list(enumerate(input_shape[1:], start=1))))
        
        self.kernel = self.add_weight(name='kernel', 
                                      shape=input_shape[1:],
                                      initializer='uniform',
                                      constraint=self.kernel_constraint,
                                      trainable=True)
        super(SinglyConnected, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return np.multiply(x,self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape)

df_classes = pd.read_csv('groupings-csv/canidae_Imagenet.csv', usecols=['wnid'])
classes = sorted([i for i in df_classes['wnid']])
whole_list = os.listdir(imagenet_test)
oc_classes = sorted([i for i in whole_list if i not in classes])

imagenet_test = '/mnt/fast-data16/datasets/ILSVRC/2012/clsloc/val/'
ImageGen = ImageDataGenerator(fill_mode='nearest',
                              horizontal_flip=True,
                              rescale=None,
                              preprocessing_function=preprocess_input,
                              data_format="channels_last",
                              )

df_classes = pd.read_csv('groupings-csv/canidae_Imagenet.csv', usecols=['wnid'])
classes = sorted([i for i in df_classes['wnid']])

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
  eval_df = pd.DataFrame(columns = ['model_name', 'ic_loss', 'oc_loss', 'whole_loss', 'ic_acc', 'oc_acc', 'whole_acc'])
  model_path_list = os.listdir(model_file)
  for i in range(len(model_path_list)):
    model_path = model_path_list[i]
    model = load_model(model_file+model_path, custom_objects={'SinglyConnected': SinglyConnected})
    ic_loss, ic_acc = model.evaluate_generator(in_context_generator, in_context_steps, verbose=1)
    oc_loss, oc_acc = model.evaluate_generator(out_context_generator, out_context_steps, verbose=1)
    whole_loss, whole_acc = model.evaluate_generator(whole_set_gen, whole_set_steps, verbose=1)
    eval_df.loc[i] = {'model_name': model_path[:-3], 
                      'ic_loss': ic_loss, 
                      'oc_loss': oc_loss, 
                      'whole_loss': whole_loss, 
                      'ic_acc': ic_acc, 
                      'oc_acc': oc_acc, 
                      'whole_acc': whole_acc}

  return eval_df




model_file = 'canidae/'
eval_df = auto_evaluate(model_file)
print (eval_df)
eval_df.to_csv('single_att_results/canidae.csv')