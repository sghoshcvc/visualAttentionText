# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 17:36:27 2016

@author: sgnosh
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 19:32:37 2016

@author: sgnosh
"""
import cPickle as pkl
import string
import numpy as np
import sys
import trie_edit_dist as tr

def get_lang_prob(sample,trie):
    
    # load language model probs
#    with open('lmProb.pkl', 'rb') as f:
#        lmProb = pkl.load(f)
#    alphabets =list(string.ascii_lowercase)
#    digits = list(string.digits)
#    alphabets.extend(digits)
    with open('dictionary.pkl', 'rb') as f:
        worddict = pkl.load(f)
    word_idict = dict()
    for kk, vv in worddict.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'
    
    
    #alphabets.append(' ')
    lpr=np.ones((len(sample),38))*sys.float_info.min
    k=0
    i=0
    if not sample:
        for cc,chars in word_idict.iteritems():
            if chars != ' 'or chars !='unk' or chars != '<eos>':
                key = ' '+chars
                x=lmProb[1][key]/ float(88172)
                lpr.append(x)
    else:
        for samples in sample:
            k=0
            if not samples: # level 1 of trie
                 wts={key:trie.children[key].weight for key in trie.children }
                 sumWts = sum(wts.itervalues())
                 for c,chars in word_idict.iteritems():
#                     if c>35:
#                         continue
                     if chars != ' ' and chars !='UNK' and chars != '<eos>':
                         key = ' '+chars
                         try:
                             lpr[0][c]=wts[chars]/ float(sumWts)
                         except:
                             lpr[0][c] =sys.float_info.min
                                 
                         #k=k+1
                         #lpr[0][k]=(x)
            else:
                
                prefix= ''.join(word_idict[x] for x in samples)
                node= trie.getNode(prefix)
                wtNode = node.weight
                wts={key:node.children[key].weight for key in node.children }
#                if len(prefix)>3:
#                    prefix=prefix[len(prefix)-3:]
                for c,chars in word_idict.iteritems():
                   # if c>35 or c==1:
                    if c==1:    
                        continue
                    if c==0:
                        chars='\n'
                    key = prefix+chars
                    try:
                        lpr[i][c] = wts[chars]/float(wtNode) #lmProb[len(key)-1][key]/max(float(lmProb[len(key)-2][key[0:-1]]),1)
                    except:
                        lpr[i][c] =sys.float_info.min
                    k=k+1
            i=i+1        
             #lpr.append(p)
    return lpr
        #lpr.append(
            
    # first step generate probability of ' (a-z|0-9)'
    
        
#    else:        
#        for samples in sample:
#            key='ab'
#            lmProb[1][key]/float(lmProb[0][key[0]])
#    for chars in alphabets:
           