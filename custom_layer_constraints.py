# -*- coding: utf-8 -*-
"""
Customized singly connected layer

"""

import keras
from keras.models import Sequential, Model
from keras import backend as K
from keras.layers import Layer
import numpy as np

#Customize a constraint class that clip w to be [K.epsilon(), inf]
from keras.constraints import Constraint

class CustomConstraint (Constraint):
    
    def __call__(self, w):
        new_w = K.clip(w, K.epsilon(), None)
        return new_w

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

