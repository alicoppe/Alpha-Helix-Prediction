import numpy as np
import ast
import os
import time

start_time = time.perf_counter()
# Added these two lines of code because relative paths were not working for me
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)


def readInput(inputFile):
    sequences = []
    names = []
    seq_strings = []
    with open(inputFile, 'r') as f:
        while True:
            
            name = f.readline()
            sequence = f.readline()
            
            if not sequence: break
            
            seq_strings.append(sequence[:-1])
            sequences.append(one_hot_encode_amino_acids(sequence[:-1]))
            names.append(name)
    
    
    return sequences, names, seq_strings

def one_hot_encode_amino_acids(input_string):
    # Takes in input string and returns an array of one hot encoding for each amino acid
    
    amino_acid_mapping = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}

    one_hot_array = np.zeros((len(input_string), len(amino_acid_mapping)))
    
    for i in range(len(input_string)):
        try:
            one_hot_array[i, [amino_acid_mapping[input_string[i]]]] = 1
        except:
            one_hot_array = one_hot_array[:-1]
        
    return one_hot_array

def decode_helices(array):
    # Returns a string from a numpy array of 0's and 1's for the output file
    string = ''
    
    for i in range(array.shape[0]):
        if array[i] == 1:
            string += 'H'
        else:
            string += '-'
            
    return string

def create_windows_dataset(sequences, window_size=11):
    # Takes a list of numpy arrays of one hot encoded amino acids, and returns a numpy array of a specific window
    # size to classify each amino acid
    pad_length = (window_size-1)//2
    
    dataset_sequences = []
    
    for i in range(len(sequences)):
        seq = np.pad(sequences[i], ((pad_length, pad_length), (0, 0)))
        
        for j in range(sequences[i].shape[0]):
            dataset_sequences.append(seq[j:j+window_size])

    return np.array(dataset_sequences)

def softmax(logits):
    # Takes the softmax of categorical data
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return probabilities

def GNB_probabilities(input_array, feature_log_prob, class_log_prior):
    log_likelihoods = np.dot(input_array, feature_log_prob.T) + class_log_prior
    output = (softmax(log_likelihoods))[:, 1] # Take the softmax of the log likelihoods, and take the probability of alpha helix, which is column 1
    # Since this is only binary classification, but the trained model on sklearn made predictions as 2 classes 
    return output
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)


def lstm_cell(X_t, h_prev, c_prev, W, U, b, units=64):
    # Weight matrix of current LSTM unit
    W_i = W[:, :units] # input gate
    W_f = W[:, units: units * 2] # forget gate 
    W_c = W[:, units * 2: units * 3] # cell state
    W_o = W[:, units * 3:] # forget gate

    # Weight matrix connecting previous LSTM unit to current
    U_i = U[:, :units]
    U_f = U[:, units: units * 2]
    U_c = U[:, units * 2: units * 3]
    U_o = U[:, units * 3:]

    # Bias vectors 
    b_i = b[:units]
    b_f = b[units: units * 2]
    b_c = b[units * 2: units * 3]
    b_o = b[units * 3:]
    
    # Input gate
    i_t = sigmoid(X_t @ W_i + h_prev @ U_i + b_i)
    
    # Forget gate
    f_t = sigmoid(X_t @ W_f + h_prev @ U_f + b_f)
    
    # Cell state
    c_t = f_t * c_prev + i_t * tanh(X_t @ W_c + h_prev @ U_c + b_c)
    
    # Output gate
    o_t = sigmoid(X_t @ W_o + h_prev @ U_o + b_o)
    
    # Hidden state
    h_t = o_t * tanh(c_t)
    
    return h_t, c_t

def dense_layer(input_data, w_dense, b_dense):
    # Dense layer of the model
    
    output = input_data@w_dense + b_dense
    return output

def forward_propagation(sequence,  W_lstm, U_lstm, b_lstm, W_dense, b_dense, units=64):
    # Loops through every value of a sequence using the lstm model and predicts an ouput
    
    c_t = np.zeros((1, units))
    h_t = np.zeros((1, units))
    
    preds = np.zeros((len(sequence)))
    for i in range(sequence.shape[0]):
        h_t, c_t = lstm_cell(sequence[i,:], h_t, c_t, W_lstm, U_lstm, b_lstm)
        
        value = dense_layer(h_t, W_dense, b_dense)

        binary_output = 1 if value > 0.5 else 0
        
        preds[i] = binary_output
        
    return preds

testingFile= '../input_file/infile.txt'
parameter_file = 'parameters.txt'

with open(parameter_file, 'r') as f:
    # Naive Bayes log likelihoods
    class_log_prior = np.array(ast.literal_eval(f.readline()[:-1]))
    feature_log_prob = np.array(ast.literal_eval(f.readline()[:-1]))
    
    # LSTM weights
    W_lstm = np.array(ast.literal_eval(f.readline()[:-1]))
    U_lstm = np.array(ast.literal_eval(f.readline()[:-1]))
    b_lstm = np.array(ast.literal_eval(f.readline()[:-1]))
    
    # Dense layer weights
    W_dense = np.array(ast.literal_eval(f.readline()[:-1]))
    b_dense = np.array(ast.literal_eval(f.readline()[:-1]))


sequences, names, seq_strings = readInput(testingFile)

# Converts the data to be analyzed by Naive Bayes
X_NB = create_windows_dataset(sequences)
X_NB = X_NB.reshape(X_NB.shape[0], X_NB.shape[1]*X_NB.shape[2])

# Calculates probabilities using the model log likelihoods from parameter file
X_NB = GNB_probabilities(X_NB, feature_log_prob, class_log_prior)
    
X_NB = X_NB.reshape(-1, 1)

# Sequences is a list of numpy arrays, consisting of one hot encoding of each amino acid
# Here we tag on the probabilities calculated from Naive Bayes onto each amino acid of the one hot encoded sequences
start = 0
for i in range(len(sequences)):
    length = sequences[i].shape[0]
    sequences[i] = np.concatenate((sequences[i], X_NB[start: start+length, :]), axis=1)
    
    start += length

# Now for every sequence we run through forward propagation and which returns an array named preds
# that contains 0s or 1s corresponding to no helix or helix, then we use decode_helices to convert this to a string of form '---H-HH'
string_predictions = []     
for i in range(len(sequences)):
    preds = forward_propagation(sequences[i], W_lstm, U_lstm, b_lstm, W_dense, b_dense)
    string_predictions.append(decode_helices(preds))

def writingOutput(names, sequences, predictions, outputFile):
  with open(outputFile, 'w') as f:
    for i in range(len(names)):
      f.write(names[i])#+"\n")
      f.write(sequences[i]+"\n")
      f.write(predictions[i]+"\n")
  return

end_time = time.perf_counter()
print(f'Elapsed Time: {round(end_time - start_time, 3)} s')

# Write everything to the output file
outputFile="../output_file/outfile.txt"
writingOutput(names, seq_strings, string_predictions, outputFile)
