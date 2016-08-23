#!/usr/bin/python
#Regression network for predicting retention time given the count of each amino acid in the peptide. 
import time
import theano
import theano.tensor as T
import numpy as np
import lasagne
import data_handling as data
import evaluation as evaluate
import nn_models as nn

#For plotting hist and saving to file
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pp




def run(input_type, test_file, train_file, max_len, num_epochs):    
    start = time.time()
    if input_type == 'branched':
        model = nn.ConvNNModel
    elif input_type == 'unbranched':
        model = nn.CNN_nobranch
    else:
        print "Wrong input_type"
    print "Loading data"
    te_seq_mat, te_seq_mat_mirror, teY = data.create_seq_mat( test_file, max_len, ordered=True)
    tr_seq_mat, tr_seq_mat_mirror, trY = data.create_seq_mat( train_file, max_len, ordered=True)
    
    if input_type == 'branched':
        teX = data.create_branched_set(te_seq_mat, te_seq_mat_mirror)
        trX = data.create_branched_set(tr_seq_mat, tr_seq_mat_mirror)        
    else:
        trX = np.append(tr_seq_mat, tr_seq_mat_mirror, axis=0)
        trY = np.append(trY, trY, axis=0)  
        teX = np.append(te_seq_mat, te_seq_mat_mirror, axis=0)
        teY = np.append(teY, teY, axis=0)  

    #Scaling the training and test sets according to the training set
    t_0 = min(trY)
    t_N = max(trY)
    trmeanY = np.mean(trY)
    trstdY = np.std(trY)
    trY = data.scale_Y(trY, trmeanY, trstdY) 
    teY = data.scale_Y(teY, trmeanY, trstdY)
    print "Training:", trY.shape
    print "Test:", teY.shape
        
        
    #Create network object 
    input_shape = trX[0].shape
    reg_net = model(shape=input_shape)      
  
    reg_net.setup_network()
    
    #Normal code
    print "Training model: ", model.__name__
    tr_epoch_err, te_epoch_err = reg_net.train(trX, trY, num_epochs=num_epochs, print_err=True, teX=teX, teY=teY)
    print "Training time Time:"
    print time.time() - start

    
    #Predicting with weights from last iteration
    predictions, targets = reg_net.predict(teX, teY)
    
    #Scaled results to minutes
    predictions = predictions * trstdY + trmeanY
    targets = targets * trstdY + trmeanY
    
    tr_epoch_err = tr_epoch_err * trstdY
    te_epoch_err = te_epoch_err * trstdY
    
    rmse = evaluate.rmse(predictions, targets)
    l_bound, u_bound= evaluate.conf_int(predictions, targets)#, t_0, t_N)
    rel_int = evaluate.rel_conf_int(predictions, targets, alpha=0.025, t_0=t_0, t_N=t_N)
    
    
    print "Testing results:"
    print "RMSE:", rmse
    rel_err = evaluate.relative_error(predictions, targets, t_0, t_N)
    print "t_0, t_N, Relative error", t_0, t_N, rel_err
    print "95 confidence interval:", l_bound, u_bound
    print evaluate.correlation(predictions, targets)
    print "average relative 95 confidence interval:", rel_int
    return predictions, targets

    
  


