#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 23:32:24 2019

@author: yixiaowan
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import math

ctgry_list = ['ave', 'canidae', 'cloth', 'felidae', 'kitchen', 'land_trans']
imagenet_train = '/mnt/fast-data16/datasets/ILSVRC/2012/clsloc/train/'

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = numpy.average(values, weights=weights)
    # Fast and numerically precise:
    var = numpy.average((values-average)**2, weights=weights)
    variance = var * len(weights)/(len(weights) - 1)
    return (average, math.sqrt(variance))


for ctgry in ctgry_list:
    class_csv_path = 'groupings-csv/' + ctgry + '_Imagenet.csv'
    df_classes = pd.read_csv(class_csv_path)
    df_classes["number_of_img"] = ""
    classes = [i for i in df_classes['wnid']]
    acc = []
    nimg_list = []
    for class_wid in classes:
        class_path = imagenet_train + class_wid
        n_img = len(os.listdir(class_path))
        nimg_list.append(n_img)
        #df_classes[df_classes['wnid'] == class_wid]['number_of_img'] = n_img
        acc = df_classes[df_classes['wnid']== class_wid]['base_accuracy'].values[0]
        acc.append(acc)
    acc = np.array(acc)
    nimg_list = np.array(nimg_list)
    #print ( weighted_acc)
    total_img = nimg_list.sum()  
    weights = nimg_list/total_img
    #print (total_img)
    avg, std = weighted_avg_and_std(acc, weights)
    print ('The total number of images for ' + ctgry + ':' + str(total_img))
    print ('The mean of acc for ' + ctgry + ':' + str(avg))
    print ('The std of acc for ' + ctgry + ':' + str(std))
    
    
    
        
    
    
    
    


