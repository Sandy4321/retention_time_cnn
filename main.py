import sys
import argparse
import run_model as run
import data_handling as data

def parse_arguments(command_line_arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file',
                        required=True,
                        help="path to file with test data, in format: SEQ TIME ")
    parser.add_argument('--train_file',
                        required=True,
                        help="path to file with training data, in format: SEQ TIME")
    parser.add_argument('--input_type',
                        default='branched',
                        help="Input type: 'branched' or 'unbranched', '25_b'- branched network with 25 column input matrix is set as default") 
    parser.add_argument('--max_len',
                        default=25,
                        help="Maximum length of peptides, 25 is default")
    return parser.parse_args(command_line_arguments)


#python test_parameters --first --last
def main(command_line_arguments):
    args = parse_arguments(command_line_arguments)

    test_file = args.test_file
    train_file = args.train_file
    input_type = args.input_type
    max_len = int(args.max_len)
    
    #Setting number of epochs for training of network
    num_epochs = 50
    predictions, targets = run.run(input_type, test_file, train_file, max_len, num_epochs)
    ## Save predictions to file
    data.save_f(predictions, '../cnn_predictions.txt')
    

        



if __name__ == '__main__':
    main(sys.argv[1:])

