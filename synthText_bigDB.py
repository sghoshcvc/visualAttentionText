# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 17:32:47 2016

@author: sgnosh
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 09:13:32 2016

@author: sgnosh
"""

import cPickle as pkl
import gzip
import os
import sys
import time
import h5py

import numpy
from scipy.sparse import csr_matrix

experimentPrefix = '.exp9'

def save_sparse_csr(filename,array):
    numpy.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = numpy.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])


def prepare_data(caps, features, worddict, maxlen=None, n_words=10000, zero_pad=False,startIndx =0):
    """ Formats the features/data
    """
    seqs = []
    feat_list = []
    for cc in caps:
        seqs.append([worddict[w] if worddict[w] < n_words else 1 for w in cc[0]])
        feat_list.append(features[cc[1]-startIndx])

    lengths = [len(s) for s in seqs]

    if maxlen != None:
        new_seqs = []
        new_feat_list = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, feat_list):
            if l < maxlen:
                new_seqs.append(s)
                new_feat_list.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        feat_list = new_feat_list
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None, None

    y = numpy.zeros((len(feat_list), feat_list[0].shape[0])).astype('float32')
    for idx, ff in enumerate(feat_list):
        y[idx,:] = numpy.array(ff)
    y = y.reshape([y.shape[0], 4*13, 512])
    if zero_pad:
        y_pad = numpy.zeros((y.shape[0], y.shape[1]+1, y.shape[2])).astype('float32')
        y_pad[:,:-1,:] = y
        y = y_pad

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)+1

    x = numpy.zeros((maxlen, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen, n_samples)).astype('float32')
    for idx, s in enumerate(seqs):
        x[:lengths[idx],idx] = s
        x_mask[:lengths[idx]+1,idx] = 1.

    return x, x_mask, y

def load_data(load_train=False, load_dev=False, load_test=False, train_idx=[],val_idx=[],test_idx=[],path='data/synthTextMod/'):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here IMDB)
    '''

    #############
    # LOAD DATA #
    #############

    print '... loading data'

    train = None
    valid = None
    test = None

    if load_train:
        with open(path+'synthText_align.train' + experimentPrefix + '.pkl', 'rb') as f:
            train_cap = pkl.load(f)
            #train_cap=numpy.array(train_cap)
        train_cap=train_cap[train_idx[0]:train_idx[1]]
        f = h5py.File(path + 'synthText_feature.train' + experimentPrefix + '.h5','r')
        train_feat = f['feature'][train_idx[0]:train_idx[1]]
        f.close()
        #train_feat = load_sparse_csr(path+'synthText_feature.train' + experimentPrefix + '.npz')
        train = (train_cap, train_feat)

    if load_dev:
        with open(path+'synthText_align.val' + experimentPrefix + '.pkl', 'rb') as f:
            dev_cap = pkl.load(f)
        dev_cap=dev_cap[val_idx[0]:val_idx[1]]
        f = h5py.File(path + 'synthText_feature.val' + experimentPrefix + '.h5','r')
        dev_feat = f['feature'][val_idx[0]:val_idx[1]]
        f.close()
            #dev_feat = pkl.load(f)
        valid = (dev_cap, dev_feat)

    if load_test:
        with open(path+'synthText_align.test' + experimentPrefix + '.pkl', 'rb') as f:
            test_cap = pkl.load(f)
        test_cap= test_cap[test_idx[0]:test_idx[1]]
        f = h5py.File(path + 'synthText_feature.test' + experimentPrefix + '.h5','r')
        test_feat = f['feature'][test_idx[0]:test_idx[1]]
        f.close()
           # test_feat = pkl.load(f)
        test = (test_cap, test_feat)

    with open(path+'dictionary.pkl', 'rb') as f:
        worddict = pkl.load(f)

    return train, valid, test, worddict
    
def load_cap(load_train=True, load_dev=True, load_test=True, path='data/synthTextMod/'):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here IMDB)
    '''

    #############
    # LOAD DATA
    # For Big DB only caption is loaded into memory
    #############
    

    train = None
    valid = None
    test = None

    if load_train:
        with open(path+'synthText_align.train' + experimentPrefix + '.pkl', 'rb') as f:
            train_cap = pkl.load(f)
        #train_feat = load_sparse_csr(path+'synthText_feature.train' + experimentPrefix + '.npz')
        train = train_cap

    if load_dev:
        with open(path+'synthText_align.val' + experimentPrefix + '.pkl', 'rb') as f:
            dev_cap = pkl.load(f)
            #dev_feat = pkl.load(f)
        valid = dev_cap

    if load_test:
        with open(path+'synthText_align.test' + experimentPrefix + '.pkl', 'rb') as f:
            test_cap = pkl.load(f)
            #test_feat = pkl.load(f)
        test = test_cap

    with open(path+'dictionary.pkl', 'rb') as f:
        worddict = pkl.load(f)

    return train, valid, test, worddict
