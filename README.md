# TextProposals

## 1. Compilation

Requires: pyCaffe and Theano

The path for caffe needs to be provided in the file `` <model\.paths> ``  

## 2. Download the DictNet pre-trained CNN model, protofile for dictnet and LSTM weights and parameters



## 3. Execute

Executing the following command: 

 `` ./eval_memes_v1  <filename> <outputfile> ``

process a single image (`` <filename> ``) and writes to  `` <outputfile> `` the list of recognized words, one per line, with the format: 

  `` x,y,w,h,pr, <transcript> ``

where x,y,w,h define a bounding box, pr is the score for the word, and `` <transcription> `` is the recognized word.

