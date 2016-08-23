Run main.py to train a neural network and to predict retention times for a test set.

Run with: python main.py --test_file=test.txt --train_file=train.txt
optional inputs: --max_len - setting the number of columns of the matrices and the maximum peptide length that the model can handle.
                 --input_type - 'branched' or 'unbranched' network

Files for training and test set (--test_file=, --train_file=) need to have the following format without header:
DRGNMSNTEYRK 42.3
VVVTEGSLDGPVILEQK 197.7
SPLDGSFDTSNLK 167.2
VDATPEESK 42.1

The neural network models are found in nn_model.py.
