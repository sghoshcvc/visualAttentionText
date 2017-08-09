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
#import h5py
#import trie_edit_dist as tr
#import string

#from sys import stdout
#import scipy
#import  cPickle as pickle
import numpy as np
#import matplotlib.pyplot as plt
#matplotlib inline
import sys
#sys.path.insert(0, caffe_root + 'python')
caffe_root = '/home/sgnosh/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe
#plt.rcParams['figure.figsize'] = (10, 10)
#plt.rcParams['image.interpolation'] = 'nearest'
#plt.rcParams['image.cmap'] = 'gray'
#import os
#import pandas as pd
#import nltk
from set_param_paths import set_param_paths
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
        return [sample[i] for i in sidx]
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
    #trie = tr.TrieNode()
#    #WordCount=0
#    for word in open(DICTIONARY, "rt").read().split():
#        word = string.lower(word)
#        
#        WordCount += 1
#        trie.insert( word )
#
#    print "Read %d words" % WordCount
#
#    def _gencap(cc0):
#        sample, score = gen_sample(tparams, f_init, f_next, cc0, options,
#                                   trng=trng, k=k, maxlen=200, stochastic=False,alpha=0.0)
#        # adjust for length bias
#        if normalize:
#            lengths = numpy.array([len(s) for s in sample])
#            score = score / lengths
#        sidx = numpy.argsort(score)
#        return [sample[i] for i in sidx]
#    seq = _gencap(context)

    return (f_init,f_next,tparams,trng)


# single instance of a sampling process
#def gen_model(queue, rqueue, pid, model, options, k, normalize, word_idict, sampling):
#    import theano
#    from theano import tensor
#    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
#
#    trng = RandomStreams(1234)
#    # this is zero indicate we are not using dropout in the graph
#    use_noise = theano.shared(numpy.float32(0.), name='use_noise')
#    DICTIONARY ="lexicon.txt"
#    # get the parameters
#    params = init_params(options)
#    #model = model+estll.npz'
#    params = load_params(model, params)
#    tparams = init_tparams(params)
#    
#    trielex = tr.TrieNode()
#    WordCount=0
#    for word in open(DICTIONARY, "rt").read().split():
#        word = string.lower(word)
#        if word.isalnum:
#        
#            WordCount += 1
#            trielex.insert( word )
#
#    print "Read %d words" % WordCount
#    # build the sampling computational graph
#    # see capgen.py for more detailed explanations
#    f_init, f_next = build_sampler(tparams, options, use_noise, trng, sampling=sampling)
#    smallLexFile = "SVT_lex.txt" 
#    with open(smallLexFile,'r') as sflex:931 18 45 31
#         slex = sflex.readlines()
#    smallVocab =[line.rstrip('\n') for line in slex] 
###with open(annotationFile,'r') as fann:
## #   ann = fann.readlines()
#    
#
#    print "Read %d words" % WordCount
#
#    def _gencap(cc0,idx):
#        trie = tr.TrieNode()
#        WordCount=0
#        vocab = smallVocab[idx].split(',')
#        for word in vocab:
#            word = string.lower(word)
#            if word.isalnum:
#        
#                WordCount += 1
#                trie.insert( word )
#        sample, score = gen_sample(tparams, f_init, f_next, cc0, options,
#                                   trng=trng, k=k, maxlen=200, stochastic=False,alpha=0.25)
#        # adjust for length bie
#        if normalize:
#            lengths = numpy.array([len(s) for s in sample])
#            score = score / lengths
#        sidx = numpy.argsort(score)
#        return [sample[i] for i in sidx]
#
#    while True:
#        req = queue.get()
#        # exit signal
#        if req is None:
#            break
#
#        idx, context = req[0], req[1]
#        print pid, '-', idx
#        seq = _gencap(context,idx)
#        rqueue.put((idx, seq))
#
#    return 

#def load_TestCap(path,saveto):
#    annotation_path = path+saveto+'_annotate.txt'
#    if saveto == 'COCO':
#        with open(annotation_path,'r') as fp:
#            x=fp.readlines()
#        test =[files.split('\t')[1] for files in x]
#    else:e some ananconda's
#        annotations = pd.read_table(annotation_path, sep=' ', header=None,names=['image', 'caption'])
#        captions = annotations['caption'].values
#        captions[pd.isnull(captions)]='nan'
#        #imageList = annotations['image'].values
#        
#                #test_feat = pkl.load(f)
#        test = captions
#    
##    annotations = pd.read_table(annotation_path, sep='\t', header=None,names=['indx','caption','image'])
##    captions = annotations['caption'].values
##    captions[pd.isnull(captions)]='nan'
#    #imageList = annotations['image'].values
#    
#           # test_feat = pkl.load(f)
##    test = captions
#
#    with open(path+'dictionary.pkl', 'rb') as f:
#        worddict = pkl.load(f)
#
#    return test, worddict
#
#def load_TestData(path,experimentPrefix,saveto,minIdx,maxIdx):
#   print '... loading data'
#
#   
#   test = None
#
#   annotation_path = path+saveto+'_annotate.txt'
#   if saveto == 'COCO':
#       with open(annotation_path,'r') as fp:
#           x=fp.readlines()
#       
#       captions =[files.split('\t')[1] for files in x] 
#   else:export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
#       annotations = pd.read_table(annotation_path, sep=' ', header=None,names=['image', 'caption'])
#       captions = annotations['caption'].values
#       captions[pd.isnull(captions)]='nan'
#       #test = captions
#
#    #imageList = annotations['image'].values
#    
#            #test_feat = pkl.load(f)
#
##   annotations = pd.read_table(annotation_path, sep='\t', header=None,names=['indx','caption','image'])
##   captions = annotations['caption'].values
##   captions[pd.isnull(captions)]='nan'
#    #imageList = annotations['image'].values
#    
#            #test_feat = pkl.l"oad(f)
#   test_cap = captions
#  
#   f = h5py.File(path + saveto+'_Text_feature.test' + experimentPrefix + '.h5','r')
#   test_feat = f['feature'][minIdx:maxIdx]
#   f.close()
#           # test_feat = pkl."oad(f)
#   test = (test_cap, test_feat)
#
# 
#
#   return test

def set_up_caffe(caffe_root,protoFile,modelFile,batchSize,useCPU):
    #Setup
#    print 'Start Setup'
#    sys.path.insert(0, caffe_root + 'python')
#    import caffe 
    ##originalImagesPath = 'data/coco/originalImages'
    #trainImagesPath = '/synthText/alvin/dataset/MSCOCO2014/train2014_224/'
    #valImagesPath = '/tmp3/alvin/dataset/MSCOCO2014/val2014_224/'
    caffe_root = caffe_root #'/home/sgnosh/caffe/'
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
#    caffe.set_device(0)
#    caffe.set_mode_gpu()
    #caffe.set_mode_gpu([0])
    if useCPU:
        caffe.set_mode_cpu()
    else:
        caffe.set_device(0)
        caffe.set_mode_gpu()
#    #caffe.set_device(0)
    net = caffe.Net(mjLayoutFilename,mjModelFile,caffe.TEST)
    
## input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    net.blobs['data'].reshape(batchSize, 1, 32, 100)
    return net,transformer

def read_image(imagePath,fn,bbox):
    img = imio.imread(imagePath+fn)
    img = img[bbox[1]:bbox[1]+bbox[3]-1,bbox[0]:bbox[0]+bbox[2]-1,:]
    if len(img.shape) == 3 and img.shape[2] == 3:
        img =np.around(np.dot(img[...,:3], [0.2989, 0.5870, 0.1140]))
        img = np.array(img,dtype=np.uint8)
    img = resize(img, (32,100), order=1, preserve_range=True)
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
def main(image_name, saveto, k=1, normalize=False, zero_pad=False,sampling=False, pkl_name=None):
    # load model model_options
#    if pkl_name is None:
#        pkl_name = model
#    with open('%s.pkl'% pkl_name, 'rb') as f:
#        options = pkl.load(f)
#    caffe_root = '/home/sgnosh/caffe/'
#    image_path = 'images/' #image_path #'/media/sgnoshdef _gencap(cc0):       
#    protoFile = 'models/synthText/dictnet_vgg_deploy.prototxt'
#    modelFile = 'models/synthText/dictnet_vgg.caffemodel'
#    batchSize = 1
    # setting up caffe here ---euracat
    caffe_root,image_path,protoFile,modelFileCaffe,batchSize,modelFileLSTM,dictFile,paramFile,useCPU = set_param_paths()
    net,transformer =set_up_caffe(caffe_root,protoFile,modelFileCaffe,batchSize,useCPU)
    if paramFile is None:
        paramFile = modelFileLSTM.replace('.npz','.pkl')
    with open('%s'% paramFile, 'rb') as f:
        options = pkl.load(f)
    

    # fetch data, skip ones we aren't using to save time
    # import pdb; pdb.set_trace()
    
    #load_data, prepare_data = get_dataset(options['dataset'])
    #_, valid, test, worddict = load_data(load_train=False, load_dev=True if 'dev' in datasets else False,
     #                                        load_test=True if 'test' in datasets else False)
    #load_cap, prepare_data,load_data = get_dataset(options['dataset'])
    #testCap,worddict = load_TestCap('/media/sgnosh/DATA/coco_image/bbIM/','COCO')
    #testCap,worddict = load_TestCap(saveto+'TestWords/',saveto)
    #test = load_TestData('svtTestWords/','.exp4')
    
    with open(dictFile, 'rb') as f:
        worddict = pkl.load(f)
    word_idict = dict()
    for kk, vv in worddict.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'
    # generate models
    f_init,f_next,tparams,trng = gen_model(modelFileLSTM, options, k, normalize, word_idict, sampling)
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
    proposals = pd.read_table(image_path+image_name.replace('.jpg','.txt'), sep=' ', header=None,names=['x', 'y','w','h','prob'])
    nProposals = proposals.shape[0]
    # for every proposal do the following
    #bbox = np.empty([4,nProposals])
    bbox = proposals.as_matrix()
#    bbox[0]=proposals['x'].values
#    bbox[1]=propsals['y'].values
#    bbox[2]=proposals['w'].values
#    bbox[3]=proposals['h'].values
#    
    caps = [None] * nProposals
   
    
    imgs = [read_image(image_path,image_name,bbox[idx,:-1]) for idx in range(nProposals)]
     # for each batch do the followinng
    
    #imgs = np.array(imgs)
    #imgs = imgs.reshape(batchSize,1,32,100)
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
        startime = time.clock()
        for idx in range(ctx.shape[0]):
            # calculate feature for every proposal here 
            
            #imName = '=sample1.jpg'
            st = time.clock()
           # img = read_image(image_path,image_name,bbox[idx,:-1])
            
            
            cc = (ctx[idx]).reshape([4*13,512]) # as per the input feature size
            
            if zero_pad:
                        cc0 = numpy.zeros((cc.shape[0]+1, cc.shape[1])).astype('float32')
                        cc0[:-1,:] = cc
            else:
                        cc0 = cc
            resp=gencap(cc0,f_init,f_next,tparams,trng,options,k,normalize)
            caps[start+idx] = resp
            end = time.clock()
            print end-st
    textResult = _seqs2words(caps)
        # save it in a file
    res = open('result.txt','w')
    for x in textResult:
        res.write(x+'\n')
    res.close()
    end = time.clock()
    print end-startime
        
        
#    for i in xrange(0,len(testCap),1107): 
#        #test = load_TestData('/media/sgnosh/DATA/coco_image/bbIM/','.exp4',saveto,i,min(i+1000,len(testCap)))
#        test = load_TestData(saveto+'TestWords/','.exp4',saveto,i,min(i+1107,len(testCap)))
#        #trainB, validB, test, worddictB = load_data(load_test=True,test_idx=(0,len(testCap)))
#    
#        # <eos> means end of sequence (aka periods), UNK means unknown
#       
#    
#        # create processes
#        queue = Queue()
#        rqueue = Queue()
#        processes = [None] * n_process
#        for midx in xrange(n_process):
#            processes[midx] = Process(target=gen_model, 
#                                      args=(queue,rqueue,midx,model,options,k,normalize,word_idict, sampling))
#            processes[midx].start()
#    
        # index -> words
    #    def _seqs2words(caps):
    #        capsw = []
    #        for cc in caps:annotations = pd.read_table(annotation_path, sep=' ', header=None,names=['image', 'caption'])
    #            ww = []
    #            for w in cc:
    #                if w == 0:
    #                    break
    #                ww.append(word_idict[w])
    #            capsw.append(' '.join(ww))
    #        return capsw
        
       
    
#        # unsparsify, reshape, and queue
#        def _send_jobs(caps,contexts):
#            k=0
#            for idx, ctx in enumerate(contexts):
#                #if str.isalnum(caps[idx]) and len(caps[idx])>2:
#               # cc = ctx.todense().reshape([4*13,512])
#                cc = ctx.reshape([4*13,512])
#                if zero_pad:
#                    cc0 = numpy.zeros((cc.shape[0]+1, cc.shape[1])).astype('float32')
#                    cc0[:-1,:] = cc
#                else:
#                    cc0 = cc
#                    queue.put((idx, cc0))
#                    k=k+1
#    
#        # retrieve caption from process
#        def _retrieve_jobs(n_samples):
#            caps = [None] * n_samples
#            for idx in xrange(n_samples):
#                resp = rqueue.get()
#                caps[resp[0]] = resp[1]
#                if numpy.mod(idx, 10) == 0:
#                    print 'Sample ', (idx+1), '/', n_samples, ' Done'
#            return caps
#    
#        ds = datasets.strip().split(',')
#    
#        # send all the features for the various datasets
#        for dd in ds:
#            if dd == 'dev': 
#                print 'Develpment Set...'
#                _send_jobs(valid[1])
#                # caps = _seqs2words(_retrieve_jobs(len(valid[1])))
#                caps = _seqs2words(_retrieve_jobs(valid[1].shape[0]))
#                # import pdb; pdb.set_trace()
#                
#                with open(saveto+'.dev.txt', 'w') as f:
#                    print >>f, '\n'.join(caps)
#                print 'Done'
#    
#            if dd == 'test':
#                print 'Test Set...',
#                _send_jobs(test[0],test[1])
#                # caps = _se)qs2words(_retrieve_jobs(len(test[1])))
#                caps = _seqs2words(_retrieve_jobs(test[1].shape[0]))
#                # import pdb; pdb.set_trace()
#                with open(saveto+'9m.trie.test.lm'+str(indx)+'.txt', 'w') as f:
#                    print >>f, '\n'.join(caps)
#                print 'Done'
#        # end processes
#        for midx in xrange(n_process):
#            queue.put(None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
#    parser.add_argument('-k', type=int, default=1)
#    parser.add_argument('-sampling', action="store_true", default=False) # this only matters for hard attention
#    parser.add_argument('-p', type=int, default=5, help="number of processes to use")
#    parser.add_argument('-n', action="store_true", default=False)
#    parser.add_argument('-z', action="store_true", default=False)
#    parser.add_argument('-d', type=str, default='dev,test')
#    parser.add_argument('-pkl_name', type=str, default=None, help="name of pickle file (without the .pkl)")
   # parser.add_argument('-model', type=str,default = 'model_v1.0.npz')
    parser.add_argument('-image_name', type=str,default = '46_img.jpg')
#    parser.add_argument('-image_dir', type=str,default='Memes')
#    parser.add_argument('-param_file',type=str,default='synthText_deterministic_model.exp9.npz', help="name of pickle file (without the .pkl)")
    parser.add_argument('-saveto', type=str,default='46_res.txt')

    args = parser.parse_args()
   # main(args.model, args.saveto, k=args.k, zero_pad=args.z, pkl_name=args.pkl_name,  n_process=args.p, normalize=args.n, datasets=args.d, sampling=args.sampling)
    #for i in range(18):
    #i=9
    
    main(args.image_name,args.saveto) #model,args.image_dir,pkl_name=args.param_file,normalize=False)
#synthText_deterministic_model.exp9.npz_epoch_10