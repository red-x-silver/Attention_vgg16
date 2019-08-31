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


ctgry_list = ['ave', 'canidae', 'cloth', 'felidae', 'kitchen', 'land_trans']
imagenet_train = '/mnt/fast-data16/datasets/ILSVRC/2012/clsloc/train/'

for ctgry in ctgry_list:
    class_csv_path = 'groupings-csv/' + ctgry + '_Imagenet.csv'
    df_classes = pd.read_csv(class_csv_path)
    df_classes["number_of_img"] = ""
    classes = [i for i in df_classes['wnid']]
    weighted_acc = []
    for class_wid in classes:
        class_path = imagenet_train + class_wid
        n_img = len(os.listdir(class_path))
        df_classes[df_classes['wnid'] == class_wid]['number_of_img'] = n_img
        acc = df_classes[df_classes['wnid']== class_wid]['base_accuracy'].values[0]
        weighted_acc.append(acc*n_img)
    print ( weighted_acc)
    total_img = df_classes['number_of_img'].sum()  
    print (total_img)
    weighted_acc = [i/total_img for i in weighted_acc]
    weighted_acc = np.array(weighted_acc)
    print ('The total number of images for ' + ctgry + ':' + total_img)
    print ('The mean of acc for ' + ctgry + ':' + weighted_acc.sum())
    print ('The std of acc for ' + ctgry + ':' + total_img.std())
    
    
    
        
    
    
    
    


