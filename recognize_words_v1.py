# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 15:41:47 2016

@author: sgnosh
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 18:52:31 2016

@author: sgnosh
""

""
Sampling script for attention models

Works on CPU with support for multi-process
"""
import argparse
import numpy
import cPickle as pkl
import pandas as pd
import time

import numpy as np

import sys
import os


#
from set_param_paths import set_param_paths

pathDict = set_param_paths()
sys.path.insert(0, pathDict['caffe_root'] + 'python')
import caffe

from skimage.transform import resize as resize
from skimage import io as imio 

from capgen import build_sampler, gen_sample, \
                   load_params, \
                   init_params, \
                  init_tparams
                  

#from multiprocessing import Process, Queue
def gencap(cc0,f_init,f_next,tparams,trng,options,k,normalize):
        sample, score = gen_sample(tparams, f_init, f_next, cc0, options,
                                   trng=trng, k=k, maxlen=200, stochastic=False,alpha=0.0)
        # adjust for length bias
        if normalize:
            lengths = numpy.array([len(s) for s in sample])
            score = score / lengths
        sidx = numpy.argsort(score)
        return [(sample[i],score[i]) for i in sidx]
    #seq = _gencap(context)

    #return (idx, seq)
def gen_model(model, options, k, normalize, word_idict, sampling):
    import theano
    from theano import tensor
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

    trng = RandomStreams(1234)
    
   # DICTIONARY = "lexicon.txt"
    # this is zero indicate we are not using dropout in the graph
    use_noise = theano.shared(numpy.float32(0.), name='use_noise')

    # get the parameters
    params = init_params(options)
    params = load_params(model, params)
    tparams = init_tparams(params)

    # build the sampling computational graph
    # see capgen.py for more detailed explanations
    f_init, f_next = build_sampler(tparams, options, use_noise, trng, sampling=sampling)
    

    return (f_init,f_next,tparams,trng)



def set_up_caffe(protoFile,modelFile,batchSize,useCPU):
    #Setup
    #caffe_root = '/home/sgnosh/caffe/'
#    sys.path.insert(0, caffe_root + 'python')
#    import caffe 
    print 'Start Setup'
    ##originalImagesPath = 'data/coco/originalImages'
    #trainImagesPath = '/synthText/alvin/dataset/MSCOCO2014/train2014_224/'
    #valImagesPath = '/tmp3/alvin/dataset/MSCOCO2014/val2014_224/'
    #caffe_root = caffe_root #'/home/sgnosh/caffe/'
    #imagePath = image_path #'/media/sgnosh/c0907750-8b66-413b-a685-0307b5b5d0ef/mnt/ramdisk/max/90kDICT32px/'
    mjLayoutFilename = protoFile #'models/synthText/dictnet_vgg_deploy.prototxt'
    mjModelFile = modelFile #'models/synthText/dictnet_vgg.caffemodel'
    #dataPath = 'data/synthText/'
    #annotation_path = 'synthText/annotation.txt'
    #splitFileName = dataPath + 'dataset_coco.json'
    #experimentPrefix = '.exp4'
    #print 'End Setup'
    #
    #capDict = pickle.load(open('capdict.pkl','rb'))def _gencap(cc0):    
    ##Setup caffe
    #print 'Start Setup caffe'
    if useCPU:
        caffe.set_mode_cpu()
    else:
        caffe.set_device(0)
        caffe.set_mode_gpu()
    #caffe.set_mode_gpu([0])
#    caffe.set_mode_cpu()
#    #caffe.set_device(0)
    net = caffe.Net(mjLayoutFilename,mjModelFile,caffe.TEST)
    
## input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    net.blobs['data'].reshape(batchSize, 1, 32, 100)
    return net,transformer

def read_image(imagePath,filename):
   # for filename in os.listdir(imagePath):
   
   print imagePath+filename
   img =imio.imread(imagePath+filename)
   if len(img.shape) == 3 and img.shape[2] == 3:
       img =np.around(np.dot(img[...,:3], [0.2989, 0.5870, 0.1140]))
   img = np.array(img,dtype=np.uint8)
   img = resize(img, (32,100), order=1)
   img = np.array(img,dtype=np.float32) # convert to single precision
   img = (img -np.mean(img)) / ( (np.std(img) + 0.0001)/128 )
   return img
            #net.blobs['data'].data[...] = transformer.preproce931 18 45 31ss('data', img)
    
# return feature of one batch
    #input: img nd array
    #input : net caffe net object
    #input : transformer caffe transformer object
def feature_extractor(img,net,transformer):
    #net.blobs['data'].data[...] = transformer.preprocess('data', img)
            #net.blobs['data'].data[...] = transformer.preprocess('data', img)
    out = net.forward()
    feat = net.blobs['conv4'].data
    #reshapeFeat = np.swapaxes(feat,0,2)
    #reshapeFeat2 = np.reshape(reshapeFeat,(1,-1))
    reshapeFeat = np.swapaxes(feat,1,3)
    reshapeFeat2 = np.reshape(reshapeFeat,(feat.shape[0],-1))
    return reshapeFeat2
def main(saveto, k=1, normalize=False, zero_pad=False,sampling=False, pkl_name=None):
    # load model model_options
    
    # set paths parameters
   # caffe_root,image_path,protoFile,modelFileCaffe,batchSize,modelFileLSTM,dictFile,paramFile,useCPU = set_param_paths()
    # setting up caffe here ---euracat
#    sys.path.insert(0, caffe_root + 'python')
#    import caffe 
    batchSize= int(pathDict['batchSize'])
    net,transformer =set_up_caffe(pathDict['protoFile'],pathDict['modelFileCaffe'],batchSize,int(pathDict['useCPU']))

    if pathDict['paramFile'] is None:
        paramFile = pathDict['modelFileLSTM'].replace('.npz','.pkl')
    else:
        paramFile = pathDict['paramFile']
    with open('%s'% paramFile, 'rb') as f:
        options = pkl.load(f)
    

    with open(pathDict['dictFile'], 'rb') as f:
        worddict = pkl.load(f)
    word_idict = dict()
    for kk, vv in worddict.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'
    # generate models
    f_init,f_next,tparams,trng = gen_model(pathDict['modelFileLSTM'], options, k, normalize, word_idict, sampling)
    # generates words using dictionary
    def _seqs2words(caps):
            capsw = []
            for cc in caps:
                capW = []
                for cc0 in cc:
                    ww=[]
                    for w in cc0:
                        if w == 0:
                            break
                        ww.append(word_idict[w])
                    capW.append(''.join(ww))
                capsw.append(' '.join(capW))
                
            return capsw    
    
    # read the proposals
    #proposals = pd.read_table(pathDict['image_path']+image_name.replace('.jpg','.txt'), sep=' ', header=None,names=['x', 'y','w','h','prob'])
    
    #nProposals =proposals.shape[0]
    # for every proposal do the following
    #bbox = np.empty([4,nProposals])
    #bbox = proposals.as_matrix()
#    bbox[0]=proposals['x'].values
#    bbox[1]=propsals['y'].values
#    bbox[2]=proposals['w'].values
#    bbox[3]=proposals['h'].values
#    
    
    img_names = os.listdir(pathDict['image_path'])
    imgs = [read_image(pathDict['image_path'],filename) for filename in img_names]
    nProposals = len(imgs)
    caps = [None] * nProposals
    probs = [None] * nProposals
   
    
    
     # for each batch do the followinng
    
    #imgs = np.array(imgs)
    #imgs = imgs.reshape(batchSize,1,32,100)
    startime = time.clock()
    
    for start in range(0,nProposals,batchSize):
        for idx in range(start,min(start+batchSize,nProposals)):
            net.blobs['data'].data[idx-start] = transformer.preprocess('data', imgs[idx])
        #ctx = feature_extractor(imgs,net,transformer)
        out = net.forward()
        feat = net.blobs['conv4'].data
    #reshapeFeat = np.swapaxes(feat,0,2)
    #reshapeFeat2 = np.reshape(reshapeFeat,(1,-1))
        reshapeFeat = np.swapaxes(feat,1,3)
        ctx = np.reshape(reshapeFeat,(feat.shape[0],-1))
        # for last iteration discard the last rows
        if start+batchSize > nProposals:
            ctx = ctx[0:nProposals-start,:]        
        
        for idx in range(ctx.shape[0]):
            # calculate feature for every proposal here 
            
            #imName = '=sample1.jpg'
            #st = time.clock()
           # img = read_image(image_path,image_name,bbox[idx,:-1])
            
            
            cc = (ctx[idx]).reshape([4*13,512]) # as per the input feature size
            
            if zero_pad:
                        cc0 = numpy.zeros((cc.shape[0]+1, cc.shape[1])).astype('float32')
                        cc0[:-1,:] = cc
            else:
                        cc0 = cc
            resp=gencap(cc0,f_init,f_next,tparams,trng,options,k,normalize)
            # if more than one output needed change here
            resp_cap,prob=resp[0]
            caps[start+idx] = [resp_cap]
            probs[start+idx] =prob
           # end = time.clock()
            #print end-st
    textResult = _seqs2words(caps)
        # save it in a file
    results =np.column_stack([img_names,textResult])
#    if saveto is None:
#        res = open(image_name.replace('_img.jpg','_res.txt','w'))
#    else:
#        res = open(saveto,'w')
#    for i,x in enumerate(textResult):
#        res.write(bbox[i]+x+'\n')
#    res.close()
    np.savetxt(saveto,results,fmt='%s')
    #np.savetxt(image_name.replace('_img.jpg','_res.txt'),results,fmt='%s,%s,%s,%s,%s,%s')
    end = time.clock()
    print end-startime



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

   # parser.add_argument('-image_name', type=str,default = '46_img.jpg')

    parser.add_argument('-saveto', type=str,default='result.txt')
 

    args = parser.parse_args()
 
    main(args.saveto)
    #print status
#synthText_deterministic_model.exp9.npz_epoch_10