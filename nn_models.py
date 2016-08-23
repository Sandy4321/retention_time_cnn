#!/usr/bin/python

#DNN models 

import sys
import numpy as np
import random
from sklearn import preprocessing as preproc
import theano
import theano.tensor as T
import lasagne
import evaluation as evaluate

def iterate_minibatches(inputs, targets, batch_size):
    assert len(inputs) == len(targets)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]

#BASIC ANN MODEL
class NeuralNetworkModel:
    def __init__(self, shape):
        #self.shape = shape
        #self.mean = None #need mean per column..
        if len(shape) == 1:
            self.l = shape[0]
        elif len(shape)==3:
            self.nc = shape[0]
            self.h = shape[1]
            self.w = shape[2]
        elif len(shape)==4:
            print "4 dim input"
            #Branches, channels, height, width
            self.nb = shape[0]
            self.nc = shape[1]
            self.h = shape[2]
            self.w = shape[3]
        else:
            print "Wrong dimensions"
            raw_input()   

    def build_network(self, input_var=None):
        network = lasagne.layers.InputLayer(shape=(None, self.l), input_var=input_var)      
        network = lasagne.layers.DenseLayer(network, 50, nonlinearity=lasagne.nonlinearities.leaky_rectify)   
        network = lasagne.layers.DenseLayer(network, 50, nonlinearity=lasagne.nonlinearities.leaky_rectify)  
        network = lasagne.layers.DenseLayer(network, 1, nonlinearity=lasagne.nonlinearities.linear)
        return network

    def setup_network(self):
        input_var = T.matrix()
        target_var = T.matrix()
        network = self.build_network(input_var)    
        prediction = lasagne.layers.get_output(network)

        loss = lasagne.objectives.squared_error(prediction, target_var)
        loss = lasagne.objectives.aggregate( loss, mode='mean' )
        loss = loss + 1e-3 * lasagne.regularization.regularize_network_params( network, lasagne.regularization.l2 )

        params = lasagne.layers.get_all_params(network, trainable=True)
        updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)

        prediction_deter = lasagne.layers.get_output(network, deterministic=True)
        test_loss = lasagne.objectives.squared_error(prediction_deter, target_var)
        test_loss = test_loss.mean()
        self.train_fn = theano.function(inputs=[input_var, target_var], outputs=prediction, updates=updates, allow_input_downcast=True)

        self.pred_fn = theano.function(inputs=[input_var, target_var], outputs= [test_loss, prediction_deter], allow_input_downcast=True)       
        self.network = network
        
    def train(self, trX, trY, batch_size=100, num_epochs=200, print_err=False, teX=None, teY=None): 
        n = len(trY)
        tr_epoch_err = np.zeros(shape=(num_epochs,), dtype=np.float32)
        if print_err:
            te_epoch_err = np.zeros(shape=(num_epochs,), dtype=np.float32)
        
        for epoch in range(num_epochs):
            #print epoch
            tr_predictions = np.zeros(shape=(n,1), dtype=np.float32)
            targets_array = np.zeros(shape=(n,1), dtype=np.float32)
            batch_n = 0            
            for batch in iterate_minibatches(trX, trY , batch_size):
                inputs, targets = batch
                train_pred = self.train_fn(inputs, targets)

                start = batch_n*batch_size
                end = batch_n*batch_size + batch_size
                tr_predictions[start:end] = train_pred 
                targets_array[start:end] = targets 
                batch_n += 1
            tr_rmse = evaluate.rmse(tr_predictions, targets_array)
            tr_epoch_err[epoch] = tr_rmse
            if print_err:
                assert teX.any()
                assert teY.any()
                te_predictions, te_targets = self.predict(teX, teY)
                te_rmse = evaluate.rmse(te_predictions, te_targets)   
                te_epoch_err[epoch] = te_rmse
        if print_err:
            return tr_epoch_err, te_epoch_err
        else:
            return tr_epoch_err

    
    def predict(self, teX, teY, batch_size=100):
        tot_test_err = 0
        test_error_sum = 0
        test_err_sum_check = 0
        test_batches = 0
        n = len(teY)
        #print "Testing"
        predictions = np.zeros(shape=(n,1), dtype=np.float32)
        targets_array = np.zeros(shape=(n,1), dtype=np.float32)
        
        for batch in iterate_minibatches(teX, teY, batch_size):
            inputs, targets = batch
            test_loss, test_pred = self.pred_fn(inputs, targets)
            start = test_batches*batch_size
            end = test_batches*batch_size + batch_size
            predictions[start:end] = test_pred
            targets_array[start:end] = targets            
            test_batches += 1
        
        rmse = evaluate.rmse(predictions, targets_array)

        # Saving network parameters to file
        #np.savez('model_1.npz', *lasagne.layers.get_all_param_values(self.network))
        # 
        # with np.load('model.npz') as f:
        #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        return predictions, targets_array
    
    def __getstate__( self ):
        odict = self.__dict__.copy()
        del odict['predict_func']
        del odict['train_func']
        del odict['network']
        odict['network_params'] = lasagne.layers.get_all_param_values( self.network )
        return odict
        

    def __setstate__( self, dict ):
        self.__dict__.update( dict )
        self.init_network()
        lasagne.layers.set_all_param_values( self.network, self.network_params )
        del self.network_params
    
    

# This class is for instances of Net 11.3. Change build_network() in child structure to get any other structure for branched/unbranched network.
class ConvNNModel(NeuralNetworkModel):
    def setup_network(self):
        input_var = T.tensor4()
        target_var = T.matrix()
        network = self.build_network(input_var)    
        prediction = lasagne.layers.get_output(network)

        loss = lasagne.objectives.squared_error(prediction, target_var)
        loss = lasagne.objectives.aggregate( loss, mode='mean' )
        loss = loss + 1e-3 * lasagne.regularization.regularize_network_params( network, lasagne.regularization.l2 )

        params = lasagne.layers.get_all_params(network, trainable=True)
        updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)
        

        test_pred = lasagne.layers.get_output(network, deterministic=True)
        test_loss = lasagne.objectives.squared_error(test_pred, target_var)
        test_loss = test_loss.mean()
        
        self.train_fn = theano.function(inputs=[input_var, target_var], outputs=prediction, updates=updates,
                                        allow_input_downcast=True)
        self.pred_fn = theano.function(inputs=[input_var, target_var], outputs=[test_loss, test_pred],
                                       allow_input_downcast=True)     
        self.network = network
                  
    def build_network(self, input_var=None):       
        input_layer = lasagne.layers.InputLayer(shape=(None, self.nc, self.h, self.w), input_var=input_var)        
        branch1 = lasagne.layers.SliceLayer(input_layer, indices=slice(1), axis=1)
        branch2 = lasagne.layers.SliceLayer(input_layer, indices=slice(2), axis=1)

        #Branch 1
        branch1 = lasagne.layers.Conv2DLayer(branch1, 40, (4,2), nonlinearity=lasagne.nonlinearities.leaky_rectify ) 
        branch1 = lasagne.layers.Conv2DLayer(branch1, 40, (4,4), nonlinearity=lasagne.nonlinearities.leaky_rectify )        
        branch1 = lasagne.layers.DenseLayer(branch1, 50, nonlinearity=lasagne.nonlinearities.leaky_rectify)  
        branch1 = lasagne.layers.DenseLayer(branch1, 50, nonlinearity=lasagne.nonlinearities.leaky_rectify)  
        
        #Branch 2
        branch2 = lasagne.layers.Conv2DLayer(branch2, 40, (4,2), nonlinearity=lasagne.nonlinearities.leaky_rectify ) 
        branch2 = lasagne.layers.Conv2DLayer(branch2, 40, (4,4), nonlinearity=lasagne.nonlinearities.leaky_rectify )         
        branch2 = lasagne.layers.DenseLayer(branch2, 50,nonlinearity=lasagne.nonlinearities.leaky_rectify)
        branch2 = lasagne.layers.DenseLayer(branch2, 50,nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        #Merging branches
        network = lasagne.layers.ConcatLayer([ branch1, branch2 ])
        network = lasagne.layers.DenseLayer(network, 50, nonlinearity=lasagne.nonlinearities.leaky_rectify)  
        network = lasagne.layers.DenseLayer(network, 1, nonlinearity=lasagne.nonlinearities.linear)  
        return network       

    
    
#Unbranched version of ConvNNModel (Net 11.3)    
class CNN_nobranch(ConvNNModel):           
    def build_network(self, input_var=None):       
        branch1 = lasagne.layers.InputLayer(shape=(None, self.nc, self.h, self.w), input_var=input_var)        
        #Branch 1
        network = lasagne.layers.Conv2DLayer(network, 40, (4,2), nonlinearity=lasagne.nonlinearities.leaky_rectify ) 
        network = lasagne.layers.Conv2DLayer(network, 40, (4,4), nonlinearity=lasagne.nonlinearities.leaky_rectify ) 
        
        network = lasagne.layers.DenseLayer(network, 50, nonlinearity=lasagne.nonlinearities.leaky_rectify)  
        network = lasagne.layers.DenseLayer(network, 50, nonlinearity=lasagne.nonlinearities.leaky_rectify)  
        
        network = lasagne.layers.DenseLayer(network, 50, nonlinearity=lasagne.nonlinearities.leaky_rectify)
  
        #################
        network = lasagne.layers.DenseLayer(network, 1, nonlinearity=lasagne.nonlinearities.linear)  
        return network   
    