# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 17:31:25 2016

@author: sgnosh
"""

import theano
import theano.tensor as tensor

import cPickle as pkl
import numpy
import copy
import os
import time

from sklearn.cross_validation import KFold

# Import from util file
from util import zipp, unzip, itemlist, load_params, init_tparams, HomogeneousData
from optimizers import adadelta, adam, rmsprop, sgd
# Import from capgen definitions code
from capgen import get_dataset, init_params, \
    build_model, build_sampler, gen_sample, pred_probs, validate_options

"""Note: all the hyperparameters are stored in a dictionary model_options (or options outside train).
   train() then proceeds to do the following:
       1. The params are initialized (or reloaded)
       2. The computations graph is built symbolically using Theano.
       3. A cost is defined, then gradient are obtained automatically with tensor.grad :D
       4. With some helper functions, gradient descent + periodic saving/printing proceeds
"""
def train(dim_word=100,  # word vector dimensionality
          ctx_dim=512,  # context vector dimensionality
          dim=1000,  # the number of LSTM units
          attn_type='stochastic',  # [see section 4 from paper]
          n_layers_att=1,  # number of layers used to compute the attention weights
          n_layers_out=1,  # number of layers used to compute logit
          n_layers_lstm=1,  # number of lstm layers
          n_layers_init=1,  # number of layers to initialize LSTM at time 0
          lstm_encoder=False,  # if True, run bidirectional LSTM on input units
          prev2out=False,  # Feed previous word into logit
          ctx2out=False,  # Feed attention weighted ctx into logit
          alpha_entropy_c=0.002,  # hard attn param
          RL_sumCost=True,  # hard attn param
          semi_sampling_p=0.5,  # hard attn param
          temperature=1.,  # hard attn param
          patience=10,
          max_epochs=5000,
          dispFreq=100,
          decay_c=0.,  # weight decay coeff
          alpha_c=0.,  # doubly stochastic coeff
          lrate=0.01,  # used only for SGD
          selector=False,  # selector (see paper)
          n_words=10000,  # vocab size
          maxlen=100,  # maximum length of the description
          optimizer='rmsprop',
          batch_size = 16,
          valid_batch_size = 16,
          saveto='model.npz',  # relative path of saved model file
          validFreq=1000,
          saveFreq=1000,  # save the parameters after every saveFreq updates
          sampleFreq=100,  # generate some samples after every sampleFreq updates
          data_path='data',  # path to find data
          dataset='flickr8k',
          dictionary=None,  # word dictionary
          use_dropout=False,  # setting this true turns on dropout at various points
          use_dropout_lstm=False,  # dropout on lstm gates
          reload_=False,
          save_per_epoch=False): # this saves down the model every epoch

    # hyperparam dict
    model_options = locals().copy()
    model_options = validate_options(model_options)

    # reload options
    if reload_ and os.path.exists(saveto):
        print "Reloading options"
        with open('%s.pkl'%saveto, 'rb') as f:
            model_options = pkl.load(f)

    print "Using the following parameters:"
    print  model_options

    print 'Loading data'
    load_cap, prepare_data,load_data = get_dataset(dataset)
    train, valid, test, worddict = load_cap()

    # index 0 and 1 always code for the end of sentence and unknown token
    word_idict = dict()
    for kk, vv in worddict.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'

    # Initialize (or reload) the parameters using 'model_options'
    # then build the Theano graph
    print 'Building model'
    params = init_params(model_options)
    if reload_ and os.path.exists(saveto):
        print "Reloading model"
        params = load_params(saveto, params)

    # numpy arrays -> theano shared variables
    tparams = init_tparams(params)

    # In order, we get:
    #   1) trng - theano random number generator
    #   2) use_noise - flag that turns on dropout
    #   3) inps - inputs for f_grad_shared
    #   4) cost - log likelihood for each sentence
    #   5) opts_out - optional outputs (e.g selector)
    trng, use_noise, \
          inps, alphas, alphas_sample,\
          cost, \
          opt_outs = \
          build_model(tparams, model_options)


    # To sample, we use beam search: 1) f_init is a function that initializes
    # the LSTM at time 0 [see top right of page 4], 2) f_next returns the distribution over
    # words and also the new "initial state/memory" see equation
    print 'Building sampler'
    f_init, f_next = build_sampler(tparams, model_options, use_noise, trng)

    # we want the cost without any the regularizers
    # define the log probability
    f_log_probs = theano.function(inps, -cost, profile=False,
                                        updates=opt_outs['attn_updates']
                                        if model_options['attn_type']=='stochastic'
                                        else None, allow_input_downcast=True)

    # Define the cost function + Regularization
    cost = cost.mean()
    # add L2 regularization costs
    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # Doubly stochastic regularization
    if alpha_c > 0.:
        alpha_c = theano.shared(numpy.float32(alpha_c), name='alpha_c')
        alpha_reg = alpha_c * ((1.-alphas.sum(0))**2).sum(0).mean()
        cost += alpha_reg

    hard_attn_updates = []
    # Backprop!
    if model_options['attn_type'] == 'deterministic':
        grads = tensor.grad(cost, wrt=itemlist(tparams))
    else:
        # shared variables for hard attention
        baseline_time = theano.shared(numpy.float32(0.), name='baseline_time')
        opt_outs['baseline_time'] = baseline_time
        alpha_entropy_c = theano.shared(numpy.float32(alpha_entropy_c), name='alpha_entropy_c')
        alpha_entropy_reg = alpha_entropy_c * (alphas*tensor.log(alphas)).mean()
        # [see Section 4.1: Stochastic "Hard" Attention for derivation of this learning rule]
        if model_options['RL_sumCost']:
            grads = tensor.grad(cost, wrt=itemlist(tparams),
                                disconnected_inputs='raise',
                                known_grads={alphas:(baseline_time-opt_outs['masked_cost'].mean(0))[None,:,None]/10.*
                                            (-alphas_sample/alphas) + alpha_entropy_c*(tensor.log(alphas) + 1)})
        else:
            grads = tensor.grad(cost, wrt=itemlist(tparams),
                            disconnected_inputs='raise',
                            known_grads={alphas:opt_outs['masked_cost'][:,:,None]/10.*
                            (alphas_sample/alphas) + alpha_entropy_c*(tensor.log(alphas) + 1)})
        # [equation on bottom left of page 5]
        hard_attn_updates += [(baseline_time, baseline_time * 0.9 + 0.1 * opt_outs['masked_cost'].mean())]
        # updates from scan
        hard_attn_updates += opt_outs['attn_updates']

    # to getthe cost after regularization or the gradients, use this
    # f_cost = theano.function([x, mask, ctx], cost, profile=False)
    # f_grad = theano.function([x, mask, ctx], grads, profile=False)

    # f_grad_shared computes the cost and updates adaptive learning rate variables
    # f_update updates the weights of the model
    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads, inps, cost, hard_attn_updates)

    print 'Optimization'

    # [See note in section 4.3 of paper]
    

   # if valid:
    #    kf_valid = KFold(len(valid[0]), n_folds=len(valid[0])/valid_batch_size, shuffle=False)
    #if test:
     #   kf_test = KFold(len(test[0]), n_folds=len(test[0])/valid_batch_size, shuffle=False)

    # history_errs is a bare-bones training log that holds the validation and test error
    history_errs = []
    # reload history
    if reload_ and os.path.exists(saveto):
        history_errs = numpy.load(saveto)['history_errs'].tolist()
    best_p = None
    bad_counter = 0

    if validFreq == -1:
        validFreq = len(train[0])/batch_size
    if saveFreq == -1:
        saveFreq = len(train[0])/batch_size
    if sampleFreq == -1:
        sampleFreq = len(train[0])/batch_size

    uidx = 0
    membatch =12500
    estop = False
    for eidx in xrange(max_epochs):
        n_samples = 0

        print 'Epoch ', eidx
        # load first batch(CPU) in memory
        for i in range(0,len(train),membatch):
            idx = (i,min(i+membatch,len(train)))
            train_feat, valid_feat, test_feat, worddict = load_data(load_train=True,train_idx=idx)
            train_iter = HomogeneousData(train_feat, batch_size=batch_size, maxlen=maxlen)
            for caps in train_iter:
                n_samples += len(caps)
                uidx += 1
            # turn on dropout
                use_noise.set_value(1.)

            # preprocess the caption, recording the
            # time spent to help detect bottlenecks
                pd_start = time.time()
                x, mask, ctx = prepare_data(caps,
                                        train_feat[1],
                                        worddict,
                                        maxlen=maxlen,
                                        n_words=n_words,startIndx=idx[0])
                pd_duration = time.time() - pd_start

                if x is None:
                    print 'Minibatch with zero sample under length ', maxlen
                    continue

            # get the cost for the minibatch, and update the weights
                ud_start = time.time()
                cost = f_grad_shared(x, mask, ctx)
                f_update(lrate)
                ud_duration = time.time() - ud_start # some monitoring for each mini-batch

            # Numerical stability check
                if numpy.isnan(cost) or numpy.isinf(cost):
                    print 'NaN detected'
                    return 1., 1., 1.

                if numpy.mod(uidx, dispFreq) == 0:
                    print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'PD ', pd_duration, 'UD ', ud_duration

            # Checkpoint
                if numpy.mod(uidx, saveFreq) == 0:
                    print 'Saving...',

                    if best_p is not None:
                        params = copy.copy(best_p)
                    else:
                        params = unzip(tparams)
                    numpy.savez(saveto, history_errs=history_errs, **params)
                    pkl.dump(model_options, open('%s.pkl'%saveto, 'wb'))
                    print 'Done'

            # Print a generated sample as a sanity check
                if numpy.mod(uidx, sampleFreq) == 0:
                # turn off dropout first
                    use_noise.set_value(0.)
                    x_s = x
                    mask_s = mask
                    ctx_s = ctx
                # generate and decode the a subset of the current training batch
                    for jj in xrange(numpy.minimum(10, len(caps))):
                        sample, score = gen_sample(tparams, f_init, f_next, ctx_s[jj], model_options,
                                               trng=trng, k=5, maxlen=30, stochastic=False)
                    # Decode the sample from encoding back to words
                        print 'Truth ',jj,': ',
                        for vv in x_s[:,jj]:
                            if vv == 0:
                                break
                            if vv in word_idict:
                                print word_idict[vv],
                            else:
                                print 'UNK',
                        print
                        for kk, ss in enumerate([sample[0]]):
                            print 'Sample (', kk,') ', jj, ': ',
                            for vv in ss:
                                if vv == 0:
                                    break
                                if vv in word_idict:
                                    print word_idict[vv],
                                else:
                                    print 'UNK',
                        print

            # Log validation loss + checkpoint the model with the best validation log likelihood
                if numpy.mod(uidx, validFreq) == 0:
                    use_noise.set_value(0.)
                    train_err = 0
                    valid_err = numpy.zeros(len(range(0,len(valid),membatch)))
                    test_err = numpy.zeros(len(range(0,len(test),membatch)))
                    cnt=0

                    if valid:
                        for i in range(0,len(valid),membatch):
                            valIdx = (i,min(i+membatch,len(valid)))
                            trainB, valid_membatch, testB, worddictB = load_data(load_dev=True,val_idx=valIdx)
                            kf_valid = KFold(len(valid_membatch[0]), n_folds=len(valid_membatch[0])/valid_batch_size, shuffle=False)
                            valid_err[cnt] = -pred_probs(f_log_probs, model_options, worddict, prepare_data, valid_membatch, kf_valid,startIndx=valIdx[0]+50000).mean()
                            cnt=cnt+1
                    cnt=0        
                    if test:
                         for i in range(0,len(test),membatch):
                            testIdx = (i,min(i+membatch,len(test)))
                            trainB, validB, test_membatch, worddictB = load_data(load_test=True,test_idx=testIdx)
                            kf_test = KFold(len(test_membatch[0]), n_folds=len(test_membatch[0])/valid_batch_size, shuffle=False)
                            test_err[cnt] = -pred_probs(f_log_probs, model_options, worddict, prepare_data, test_membatch, kf_test,startIndx=testIdx[0]).mean()
                            cnt=cnt+1
                        #test_err = -pred_probs(f_log_probs, model_options, worddict, prepare_data, test, kf_test).mean()
                    valid_err = numpy.mean(valid_err)
                    test_err=numpy.mean(test_err)
                    history_errs.append([valid_err, test_err])

                # the model with the best validation long likelihood is saved seperately with a different name
                    if uidx == 0 or valid_err <= numpy.array(history_errs)[:,0].min():
                        best_p = unzip(tparams)
                        print 'Saving model with best validation ll'
                        params = copy.copy(best_p)
                        params = unzip(tparams)
                        numpy.savez(saveto+'_bestll', history_errs=history_errs, **params)
                        bad_counter = 0

                # abort training if perplexity has been increasing for too long
                    if eidx > patience and len(history_errs) > patience and valid_err >= numpy.array(history_errs)[:-patience,0].min():
                        bad_counter += 1
                        if bad_counter > patience:
                            print 'Early Stop!'
                            estop = True
                            break

                    print 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err

        print 'Seen %d samples' % n_samples

        if estop:
            break

        if save_per_epoch:
            numpy.savez(saveto + '_epoch_' + str(eidx + 1), history_errs=history_errs, **unzip(tparams))

    # use the best nll parameters for final checkpoint (if they exist)
    if best_p is not None:
        zipp(best_p, tparams)

    use_noise.set_value(0.)
    train_err = 0
    valid_err = 0
    test_err = 0
    valid_err = numpy.zeros(len(range(0,len(valid),membatch)))
    test_err = numpy.zeros(len(range(0,len(test),membatch)))
    cnt=0 
    if valid:
        for i in range(0,len(valid),membatch):
            valIdx = (i,min(i+membatch,len(valid)))
            trainB, valid_membatch, testB, worddictB = load_data(load_dev=True,val_idx=valIdx)
            kf_valid = KFold(len(valid_membatch[0]), n_folds=len(valid_membatch[0])/valid_batch_size, shuffle=False)
            valid_err[cnt] = -pred_probs(f_log_probs, model_options, worddict, prepare_data, valid_membatch, kf_valid,startIndx=valIdx[0]).mean()
            cnt=cnt+1
    cnt=0            
    if test:
        for i in range(0,len(test),membatch):
            testIdx = (i,min(i+membatch,len(test)))
            trainB, validB, test_membatch, worddictB = load_data(load_test=True,test_idx=testIdx)
            kf_test = KFold(len(test_membatch[0]), n_folds=len(test_membatch[0])/valid_batch_size, shuffle=False)
            test_err[cnt] = -pred_probs(f_log_probs, model_options, worddict, prepare_data, test_membatch, kf_test,startIndx=testIdx[0]).mean()
            cnt=cnt+1
    valid_err = numpy.mean(valid_err)
    test_err=numpy.mean(test_err)
                    
    print 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err

    params = copy.copy(best_p)
    numpy.savez(saveto, zipped_params=best_p, train_err=train_err,
                valid_err=valid_err, test_err=test_err, history_errs=history_errs,
                **params)

    return train_err, valid_err, test_err



if __name__ == '__main__':
    train(dataset='flickr30k')
