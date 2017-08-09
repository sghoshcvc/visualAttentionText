# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 12:34:41 2016

@author: sgnosh
"""
import pandas as pd
def set_param_paths():
    
#    caffe_root = '/home/sgnosh/caffe/'
#    image_path = '../MemeDB_TestSet/' #image_path #'/media/sgnoshdef _gencap(cc0):       
#    protoFile = '../models/dictnet_vgg_deploy_conv.prototxt'
#    modelFileCaffe = '../models/dictnet_vgg.caffemodel'
#    batchSize = 1
#    modelFileLSTM = '../models/model_v1.0.npz'
#    dictFile = '../models/dictionary.pkl'
#    paramFile = '../models/params.pkl'
#    useCPU = 0
    pathFile = '../models/.path'
    paths = pd.read_table(pathFile, sep='=', header=None,names=['var', 'path'])
    pathDict= paths.set_index('var')['path'].to_dict()

    return pathDict