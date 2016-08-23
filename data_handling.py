#!/usr/bin/python

import numpy as np
import evaluation as evaluate
import random
from sklearn import preprocessing as preproc


def read_f(filepath):
    f = open(filepath, 'r')
    var = np.load(f)
    f.close()
    return var

def save_f(var, filepath):
    f = open(filepath, 'w')
    np.save(f, var, allow_pickle=True, fix_imports=True)
    f.close()
    return



#Creates matrix representation of sequence. Returns array of matrix and array of mirror image matrix
#Generates new Y as data points exceeding max_len are removed 
def create_seq_mat( datafile, max_len, ordered=True):
    sequences = []
    ret_times = []
 
    data = open(datafile)
    for line in data:
        pieces = line.split()
        sequences.append(pieces[0])
        ret_times.append(np.float32(pieces[1]))
    
    data.close()
    Y = np.array(ret_times,dtype=np.float32)
    Y = Y.reshape((-1,1))
    
    assert len(Y) == len(sequences)
    
    amino_acids = load_aa_seq(ordered=ordered)
    
    rows = len(amino_acids)
    cols = max_len
    seq_data = []
    mirror_seq_data = []
    new_Y = []

    for i in range(len(Y)):
        seq = sequences[i]
        if len(seq) > max_len:
            continue
        new_Y.append(Y[i])
        seq_mat = np.zeros((rows,cols), dtype=np.float32)
        mirror_seq_mat = np.zeros((rows,cols), dtype=np.float32)
        mir_start = cols - len(seq) + 1
        for j in range(len(seq)):
            for i in range(len(amino_acids)):
                if seq[j] == amino_acids[i]:
                    seq_mat[i,j] = 1
                    mirror_seq_mat[i,-(mir_start+j)] = 1
                    break
        #####RESHAPING#####               
        seq_mat = seq_mat.reshape(1, rows, cols)
        mirror_seq_mat = mirror_seq_mat.reshape(1, rows, cols)
        
        seq_data.append(seq_mat)
        mirror_seq_data.append(mirror_seq_mat)
    
    seq_data = np.array(seq_data, dtype=np.float32)
    mirror_seq_data = np.array(mirror_seq_data, dtype=np.float32)
    new_Y = np.array(new_Y,dtype=np.float32)
    new_Y = new_Y.reshape((-1,1))
    
    #Saving seq_mat_arrays to file
    #path = '../new_datafiles/%d_seq_mat_%s.txt' % ( max_seq_len, set_name )
    #save_f(seq_data, path)
    
    #path = '../new_datafiles/%d_mirror_seq_mat_%s.txt' % ( max_seq_len, set_name )
    #save_f(mirror_seq_data, path)
    
    #Saving Y, updated for removed sequences
    #path = '../new_datafiles/%d_y_%s.txt' % ( max_seq_len, set_name )
    #save_f(new_Y, path)    
    return seq_data, mirror_seq_data, new_Y

def remove_dim(X):
    n = len(X)
    new_X = []
    for i in range(n):
        new_X.append(X[i][0])
    new_X = np.array(new_X, dtype=np.float32)
    return new_X  
    

def create_branched_set(X_set, X_set_mirr):
    X_set = remove_dim(X_set)
    X_set_mirr = remove_dim(X_set_mirr)
    n = len(X_set)
    new_set = [0]*n
    for ind in range(n):
        data = []
        data.append(X_set[ind])
        data.append(X_set_mirr[ind])
        new_set[ind] = data
    new_set = np.array(new_set, dtype=np.float32)
    return new_set


def scale_Y(Y, trmean, trstd):
    Y_scaled = (Y - trmean) / trstd
    return Y_scaled
    
    

def load_aa_seq(ordered=True):
    if ordered:
        # Order given by hydrophobicity from count linear regression ( in linreg_on_counts.py)
        return ['H', 'R', 'K', 'N', 'Q', 'G', 'S', 'C', 'T', 'E', 'A', 'D', 'P', 'V', 'Y', 'M', 'I', 'L', 'F', 'W']
        #return ['H', 'R', 'K', 'N', 'Q', 'G', 'S', 's', 'C', 'T', 't', 'E', 'A', 'D', 'P', 'V', 'Y', 'y', 'M', 'm', 'I', 'L', 'F', 'W']
    else:
        return ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']




    