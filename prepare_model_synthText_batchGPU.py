# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 18:52:25 2016

@author: sgnosh
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 23:54:17 2016

@author: sgnosh
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 19:38:18 2016

@author: sgnosh
"""

#import
import pdb
from sys import stdout
import scipy
import  cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline
import sys
import h5py
#sys.path.insert(0, caffe_root + 'python')
caffe_root = '/home/sgnosh/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
import os
import pandas as pd
import nltk
from skimage.transform import resize as resize
from skimage import io as imio 
#rom skimage import exposure

#Setup
print 'Start Setup'
##originalImagesPath = 'data/coco/originalImages'
#trainImagesPath = '/synthText/alvin/dataset/MSCOCO2014/train2014_224/'
#valImagesPath = '/tmp3/alvin/dataset/MSCOCO2014/val2014_224/'
caffe_root = '/home/sounak/caffe/'

imagePath = 'mnt/ramdisk/max/90kDICT32px/'
mjLayoutFilename = caffe_root+'models/synthText/dictnet_vgg_deploy_conv.prototxt'
mjModelFile = caffe_root +'models/synthText/dictnet_vgg.caffemodel'
dataPath = 'data/synthTextMod/'
annotation_path = 'synthText/annotation.txt'
#splitFileName = dataPath + 'dataset_coco.json'
experimentPrefix = '.exp9'
#print 'End Setup'
#


##Setup caffe
print 'Start Setup caffe'
caffe.set_mode_gpu()
caffe.set_device(0)
#caffe.set_mode_gpu()
#caffe.set_mode_cpu()
net = caffe.Net(mjLayoutFilename,mjModelFile,caffe.TEST)
## input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
#transformer.set_transpose('data', (2,0,1))
#transformer.set_mean('data', np.load('/tmp3/alvin/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
#transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
#transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
#print 'End Setup caffe'

#filelist:
#./splits/coco_val.txt
#./splits/coco_test.txt
#./splits/coco_train.txt

# set net to batch size of 50
# net.blobs['data'].reshape(10,3,224,224)

# divide in train val test for 15k
print 'Start middle'
files = ['train']
for fname in files:
    print fname 
    ferr = dataPath+fname+'_err.txt'
    ft=open(ferr,'w')
    
    #f = open('synthText/annotation_'+fname+'.txt')
    annotation_path = "data/synthTextMod/annotation_"+fname+'Cap'+experimentPrefix+'.txt'
    annotations = pd.read_table(annotation_path, sep='\t', header=None,names=['image', 'caption'])
    captions = annotations['caption'].values
    captions[pd.isnull(captions)]='nan'
    imageList = annotations['image'].values
    counter = 0
    #imageList = [i for i in f]
    numImage = len(imageList)
    #numImage = 10520
    #numImage =5
    #pdb.set_trace()
    #result = np.empty((numImage, 26624))
   # fileName = open(dataPath + 'synthText_feature.' + fname + experimentPrefix + '.pkl','ab')
    f = h5py.File(dataPath + 'synthText_feature.' + fname + experimentPrefix + '.h5','r+')
   # dset = f.create_dataset("feature",(numImage,26624),dtype=np.float32,compression='gzip')
    dset = f['feature']
    j=0
    end =7200000
    batchMem = 50 # number of batches before commits to disk
    batchSize=net.blobs['data'].data.shape[0]
    result = np.empty((batchSize*batchMem, 26624))
    nBatches= int(np.ceil(numImage/float(batchSize)))
    for i in range(nBatches):

        stdout.write("Batch Number %d\n" % i)
        if i>=14400:
            stdout.write("Batch Number processing %d\n" % i)
            for k in range(batchSize):
                if i*batchSize+k<numImage:
                    fn = imageList[i*batchSize+k]
    #                fn= fn[1:-2]
                    if fname=='train':
                        try:
                            img = caffe.io.load_image(imagePath+fn)
                            if len(img.shape) == 3 and img.shape[2] == 3:
                                img =np.around(np.dot(img[...,:3], [0.2989, 0.5870, 0.1140]))
                                img = np.array(img,dtype=np.uint8)
                            img = resize(img, (32,100), order=1, preserve_range=True)
                            img = np.array(img,dtype=np.float32) # convert to single precision
                            img = (img -np.mean(img)) / ( (np.std(img) + 0.0001)/128 )
                            net.blobs['data'].data[k] = transformer.preprocess('data', img)
                        except:
                            exc_type, exc_value, exc_traceback = sys.exc_info()
                            ft.write(imagePath+fn+' '+str(exc_value) +'\n')
                    else:
                      # img = caffe.io.load_image(imagePath+fn)
                        try:
                           img =imio.imread(imagePath+fn)
                           if len(img.shape) == 3 and img.shape[2] == 3:
                                #img = color.rgb2gray(img)
                                img =np.around(np.dot(img[...,:3], [0.2989, 0.5870, 0.1140]))
                                img = np.array(img,dtype=np.uint8)
                           #img =exposure.rescale_intensity(img,out_range=(0, 255))
                           img = resize(img, (32,100), order=1, preserve_range=True)
                           img = np.array(img,dtype=np.float32) # convert to single precision
                           img = (img -np.mean(img)) / ( (np.std(img) + 0.0001)/128 )
                           net.blobs['data'].data[k] = transformer.preprocess('data', img)
                        except:
        #                #net.blobs['data'].data[...] = transformer.preprocess('data', img)
                          # ft.write(imagePath+fn+'\n')
                            exc_type, exc_value, exc_traceback = sys.exc_info()	
                            ft.write(imagePath+fn+' '+str(exc_value) +'\n')
                else:
                    lastIndx =k
                    break
    		  		
            out = net.forward()
            feat = net.blobs['conv4'].data           
            reshapeFeat = np.swapaxes(feat,1,3)
            reshapeFeat2 = np.reshape(reshapeFeat,(batchSize,-1))
            counter += batchSize
            stdout.write("\r%d %d %d\n" % (counter,j*batchSize,(j+1)*batchSize))
            stdout.flush()
            result[j*batchSize:(j+1)*batchSize,:] = reshapeFeat2
            j=j+1
            #stdout.write("%d %d\n" % (j,np.mod(j,batchMem)))
            # copy the data to h5 file
#            if np.mod(j,batchMem)==0: 
#                
#                #resultSave = scipy.sparse.csr_matrix(result)
#                resultSave32 = result.astype('float32')
#                sz=resultSave32.shape[0]
#                stdout.write("store %d rows from %d to %d\n" % (sz,end,end+sz))
#                dset[end:end+sz,:]=resultSave32
#                end =end+sz
#                #pickle.dump(result,fileName,-1)
#                j=0
            if  i+1==nBatches:
                resultSave32 = result.astype('float32')
                sz=resultSave32.shape[0]
                resultSave32=resultSave32[0:(j-1)*batchSize+lastIndx,:]	
                sz=resultSave32.shape[0]
                print 'store %d from %d to %d rows' % (sz,end,end+sz)
                dset[end:end+sz,:]=resultSave32
                end =end+sz
            
        #j=0
       
#        fn = imageList[i]
#        fn= fn[1:-2]
#        if fname=='train':
#            try:
#                img = caffe.io.load_image(imagePath+fn)
#                if len(img.shape) == 3 and img.shape[2] == 3:
#                    img =np.around(np.dot(img[...,:3], [0.2989, 0.5870, 0.1140]))
#                    img = np.array(img,dtype=np.uint8)
#                img = resize(img, (32,100), order=1, preserve_range=True)
#                img = np.array(img,dtype=np.float32) # convert to single precision
#                img = (img -np.mean(img)) / ( (np.std(img) + 0.0001)/128 )
#                net.blobs['data'].data[...] = transformer.preprocess('data', img)
#            except:
#                ft.write(imagePath+fn+'\n')
#        else:
#          # img = caffe.io.load_image(imagePath+fn)
#            try:
#               img =imio.imread(imagePath+fn)
#               if len(img.shape) == 3 and img.shape[2] == 3:
#                    #img = color.rgb2gray(img)
#                    img =np.around(np.dot(img[...,:3], [0.2989, 0.5870, 0.1140]))
#                    img = np.array(img,dtype=np.uint8)
#               #img =exposure.rescale_intensity(img,out_range=(0, 255))
#               img = resize(img, (32,100), order=1, preserve_range=True)
#               img = np.array(img,dtype=np.float32) # convert to single precision
#               img = (img -np.mean(img)) / ( (np.std(img) + 0.0001)/128 )
#               net.blobs['data'].data[...] = transformer.preprocess('data', img)
#        
#            except:
#                #net.blobs['data'].data[...] = transformer.preprocess('data', img)
#               ft.write(imagePath+fn+'\n')
        
#        feat = net.blobs['conv4'].data[0]
#        reshapeFeat = np.swapaxes(feat,0,2)
#        reshapeFeat2 = np.reshape(reshapeFeat,(1,-1))
#        counter += 1
#        stdout.write("\r%d" % counter)
#        stdout.flush()
#        result[j,:] = reshapeFeat2
#        j=j+1
        
        #if i==0:
         #   result = resultSave32
        #else:
         #   result = scipy.sparse.vstack([result, resultSave32])
            
    print result.shape
    f.close()
    #fileName.close()

   # resultSave = scipy.sparse.csr_matrix(result)
   # resultSave32 = resultSave.astype('float32')
   # if fname == 'train':
    #    np.savez(dataPath + 'synthText_feature.' + fname + experimentPrefix, data=result.data, indices=result.indices, indptr=result.indptr, shape=result.shape)
    #else:
     #   fileName = open(dataPath + 'synthText_feature.' + fname + experimentPrefix + '.pkl','wb') 
      #  pickle.dump(result, fileName, -1) 
    

print 'End middle'



print 'Start end'
#np.savez(dataPath + 'coco_feature.' + fname + experimentPrefix, data=resultSave32.data, indices=resultSave32.indices, indptr=resultSave32.indptr, shape=resultSave.shape)

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices, indptr =array.indptr, shape=array.shape)

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

#capDict = pickle.load(open('capdict.pkl','rb'))

#files = ['test', 'val']
#for name in files:
#    counter = 0
#    with open(dataPath + 'synthText_feature.' + name + experimentPrefix + '.pkl','rb') as fp:
#        feat = pickle.load(fp)
#    #filenames = open('./splits/coco_'+name+'.txt')
#    annotation_path = "data/synthText/annotation_"+name+'Cap.txt'
#    annotations = pd.read_table(annotation_path, sep='\t', header=None,names=['image', 'caption'])
#    captions = annotations['caption'].values
#    captions[pd.isnull(captions)]='nan'
#    imageList = annotations['image'].values
#    cap = []
#    for imageFile in imageList:
#        imageFile = imageFile[:-2]
#        for sen in capDict[imageFile]:
#            cap.append([sen.rstrip(), counter])
#        counter += 1
#    saveFile = open(dataPath + 'synthText_align.' + name + experimentPrefix + '.pkl', 'wb') 
#    pickle.dump(cap, saveFile, protocol=pickle.HIGHEST_PROTOCOL) 
#    pickle.dump(feat, saveFile, protocol=pickle.HIGHEST_PROTOCOL)
#    saveFile.close()
#    #filenames.close()
#
#files = ['train']
#for name in files:
#    counter = 0
#    annotation_path = "data/synthText/annotation_"+name+'Cap.txt'
#    annotations = pd.read_table(annotation_path, sep='\t', header=None,names=['image', 'caption'])
#    captions = annotations['caption'].values
#    captions[pd.isnull(captions)]='nan'
#    imageList = annotations['image'].values
#    cap = []
#    for imageFile in imageList:
#        imageFile = imageFile[:-2]
#        for sen in capDict[imageFile]:
#            cap.append([sen.rstrip(), counter])
#        counter += 1
#    saveFile = open(dataPath + 'synthText_align.' + name + experimentPrefix + '.pkl', 'wb') 
#    pickle.dump(cap, saveFile, protocol=pickle.HIGHEST_PROTOCOL) 
#    saveFile.close()
#    #filenames.close()
#
##print wordsDict['Two']
##print resultSave32

#print 'Start end'
capDict = pickle.load(open('capdict.pkl','rb'))
files =  ['val','test','train']
for name in files:
    counter = 0
   # annotation_path = "data/synthText/annotation_"+name+'Cap.txt'
    annotation_path = "data/synthTextMod/annotation_"+name+'Cap'+experimentPrefix+'.txt'
    annotations = pd.read_table(annotation_path, sep='\t', header=None,names=['image', 'caption'])
    captions = annotations['caption'].values
    captions[pd.isnull(captions)]='nan'
    imageList = annotations['image'].values
    cap = []
    for imageFile in imageList:
        #imageFile = imageFile[:-2]
        for sen in capDict[imageFile]:
            cap.append([sen.rstrip(), counter])
        counter += 1
    saveFile = open(dataPath + 'synthText_align.' + name + experimentPrefix + '.pkl', 'wb') 
    pickle.dump(cap, saveFile, protocol=pickle.HIGHEST_PROTOCOL) 
    saveFile.close()
    #filenames.close()

#print wordsDict['Two']
#print resultSave32

print 'Start end'
