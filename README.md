# Alpha Helix Classification in Amino Acid Sequences Using Machine Learning

## Theoretical Basis:

To modify the amino acid sequence data for more efficient computation,
one-hot encoding was used. This is a technique used to represent
categorical data, where each input is converted to an array of size N,
where N is the total number of possible categories. For each of the
categories, a binary vector is created, where every element in the array
is set to 0 except for the index corresponding to the category, which is
set to 1. This method assumes there is no ranking amongst categories,
and each category is treated as independent from the other. One-hot
encoding is widely used in machine learning, as it prevents the model
from misinterpreting ordinal relationships amongst categories, which
works well for the given application.

**Naïve Bayes**

Naïve Bayes is the first algorithm used in this project, which is a
probabilistic algorithm based on Bayes' Theorem, which is represented as
the probability of existing beliefs (A) given some observed evidence
(B):

$$P\left( A \middle| B \right) = \ \frac{P(B)P(B|A)}{P(A)}$$

The denominator is often difficult to compute especially in complex
models, hence we can focus or proportional relationships:

$$P\left( A \middle| B \right) \propto P(B)P(B|A)$$

Now for the task of classification, given a set of categorical inputs,
we make the naïve assumption of conditional independence of inputs.
Using this assumption, we can calculate given a certain class label, the
likelihood of the certain set of input features as a product of
conditional probabilities of each individual feature:

$$P\left( A_{1},A_{2},\ldots A_{n} \middle| B \right) = P\left( A_{1} \middle| B \right) \bullet P\left( A_{2} \middle| B \right)\ldots \bullet P\left( A_{n} \middle| B \right)$$

For larger sets of features, to prevent numerical underflow for small
probability values, the product can be converted to the sum of log
probabilities:

$$\log{P\left( A_{1},A_{2},\ldots A_{n} \middle| B \right)} = logP\left( A_{1} \middle| B \right) + logP\left( A_{2} \middle| B \right) + ..\  + logP(A_{n}|B)$$

**Long Short-Term Memory (LSTM) Recurrent Neural Networks**

Recurrent Neural Networks (RNNs) are neural networks which pertain to
the analysis of sequential data. In these networks, individual vectors
are sequentially run through the network, and the output for each input
of the sequence is used as a hidden state which is used as part of the
input for the following input in the sequence. With this architecture,
this allows for elements of a sequence to have context provided from
previous elements to allow for more accurate predictions. These networks
suffer from the vanishing gradient problem though, in which the
computation of a gradient with respect to an output can get extremely
small for long sequences.

This problem is fixed by the Long Short-Term Memory network architecture
for RNNs, which introduces a memory cell, and a set of gates allowing it
to retain or discard information from previous layers. Due to this
complex architecture, LSTMs can better capture long-term dependencies in
sequential data. These are the following main components of an LSTM
network, which are showcased in the model in *Figure 1*:

1.  *Input gate (i):* Determines the information to be added to the cell
    state.

2.  *Forget gate (f):* Determines what information from the previous
    cell state to be discarded.

3.  *Cell State Update (g)*: The new information that is to be added to
    the cell state.

4.  *Cell State (c~t~):* Combination of the forget gate, input gate, and
    cell input.

5.  *Output Gate (o)*: Determines what part of the cell state to be
    added to the next hidden state.

6.  *Hidden State Output*: The information that is outputted from the
    LSTM cell.

<img width="468" alt="image" src="https://github.com/alicoppe/Alpha-Helix-Prediction/assets/96750083/7b61c1c0-2f20-49d8-b59b-4da06d62debe">

This LSTM cell structure is repeated for every vector in the sequence.
For each of the inputs of the Input Gate, Forget Gate, Cell State
Update, and Output Gate, there are trainable weights, which are the
extracted model parameters necessary for creating a forward pass for
prediction (Daniel & Austen, 2023).

## Code Explanation:

In the provided SSpred.py file, the parameters for the log likelihoods
of the Naïve Bayes model, the weights and biases for the LSTM model, and
the weights and biases for the dense layer of the model are loaded from
parameters.txt. The input data file containing the amino acid sequences
is then loaded, separating the output into lists of the names, a one-hot
encoded version of the sequences, and the string version of the
sequences. The sequences are then modified into specific window lengths
for each amino acid, then flattened so that the model can calculate
probabilities using the Naïve Bayes model with pre-trained log
likelihoods.

The probabilities from the Naïve Bayes model are then concatenated onto
each entry in the one hot encoded array in the sequence list, which is
then used as an input to the pre-trained LSTM model, containing 64
units, and a dense layer for classification. The input sequences are
then individually passed through the LSTM model, recalling the
lstm_cell() function for each amino acid, and adding the resulting numpy
array that is predicted to a list. Finally, each of the predicted numpy
arrays in the list are converted to string format, and are outputted to
a specified output file, along with their names and the string version
of their sequence.

## Discussion

Firstly, a Naïve Bayes model was tested for a range of amino acid
windows, these results are summarized in *Figure 2*. The window size
signifies the total amino acids analyzed to predict the output of a
single amino acid; a window size N would signify (N-1)/2 amino acids on
each side of the amino acid of interest which we want to predict. The
maximum predicted accuracy from this model was approximately 72%, which
was reached at a window size of 11 amino acids, indicating any increase
does not provide any additional valuable context for alpha helix
prediction. Since simpler models tend to be more generalizable, the
simplest model with the highest accuracy was taken for predictions,
which would be the amino acid window size of 11.

<img width="291" alt="image" src="https://github.com/alicoppe/Alpha-Helix-Prediction/assets/96750083/dc0141a4-551b-4fd7-8f94-719b979ed4e5">

One downside for prediction solely using Naïve Bayes, is that it assumes
independence of predictions, hence each amino acid is being predicted in
isolation using a specific window size, and not being considered as a
sequence. This provides good accuracy, but it is likely that
incorporating the context of the previous amino acids in the sequence
would provide additional accuracy in alpha helix prediction. For this
reason, I decided to incorporate the Naïve Bayes model with an LSTM
model where the sequence is considered as a whole, and the output of
previous amino acids in the sequence is considered for future
predictions. In this model I decided to formulate the sequences as
inputs of M x 21, where M is the variable length of each amino acid
sequence, and 21 is the number of possible amino acids (20) with one
additional column at the end for the predicted Naïve Bayes probability
at that specific amino acid. This way, the model could extract amino
acid related features in the sequence, related to one-hot encoding, as
well as utilize the sequence independent probabilities related to alpha
helices, determined by the Naïve Bayes model. The data was split into
0.8 for training the LSTM model, and 0.2 for the test set, to ensure
proper generalizability by minimizing overfitting on the training set.
The results are summarized in *Figure 3*, where it can be observed that
the model reaches a peak test set accuracy of about 76.7% at epoch 22,
where the test and training accuracies begin to diverge.

To ensure the improved model accuracy was not due to data leaking from
pretraining the data on a Naïve Bayes model and then using that to train
the LSTM model, the model was also tested on a separated dataset. In
this separated dataset, the data was split into 0.3 for Naïve Bayes
training, 0.5 for LSTM training, and 0.2 for testing. This way the Naïve
Bayes model trained on 0.3 of the dataset was used to make predictions
on the LSTM training data, and the testing data, which would then be
used to train and test the LSTM model using these predictions as
context, along with the one-hot encoded amino acids. The results were
very similar to those obtained on the previous model, with a maximum
validation accuracy of approximately 75.6%, though it took around 10
more epochs to converge to this value. The final model used for testing
was the first model, as it utilizes a larger proportion of the dataset
for training, which would be more beneficial for future predictions.

The model benefits from an increased accuracy, due to the combination of
context related insights from the LSTM model, considering correlations
between previous outputs, and amino acid values, and the sequence
independent insights from the Naïve Bayes model. Additionally, unlike
most other models, the LSTM architecture allows for the detection of
long-range dependencies that might lead to a resulting alpha helix. A
possible disadvantage to this model would be the complex architecture,
which makes for more difficult implementation, training, and longer time
to make predictions, with SSPred.py taking about 35 seconds to run. This
also causes for a lot more hyperparameters to tune, such as total
layers, sizes, learning rates, and input shape.

To improve upon the code, I would first look to improve upon the current
model, such as performing hyperparameter tuning, especially looking at
using multiple amino acids as an input sequence. Additionally, I would
look to the implementation of parallelization to make predictions, as
the model is quite slow to do a forward pass. Following this, I would
look at implementing more complex network architectures to get improved
accuracy. One downside to the current LSTM approach is that it only
considers the preceding inputs for the prediction of an alpha helix. To
address this, adapting the model to a bidirectional LSTM, which
considers previous and future inputs in the sequence, would likely
provide more accuracy in the predictive power of the model.

Another future consideration would be for the preparation of the amino
acid sequence data in more descriptive ways than one-hot encoding. Given
the nature of one-hot encoding, the intrinsic similarities or
dissimilarities between amino acids are not considered in the
categorization of the data, for example leucine and isoleucine are very
similar in structure and properties, hence would have similar behaviour.
To account for similarities amongst amino acids, they could be
categorized through a position specific scoring (PSSM), which utilizes
PSIBLAST and reflects information of sequence evolution, amino acid
conservation and mutation. These scoring matrices have been used in many
machine learning applications, and have shown excellent results, such as
an 80.18% accuracy for multiclass secondary structure prediction using
them as an input to a combined CNN and LSTM model (Cheng et al., 2020).

## References

Cheng, J., Liu, Y., & Ma, Y. (2020). Protein secondary structure
prediction based on integration of CNN and LSTM model. *Journal of
Visual Communication and Image Representation*, *71*, 102844--102844.
https://doi.org/10.1016/j.jvcir.2020.102844

‌Daniel, J., & Austen, J. (2023). *Speech and Language Processing
Sequence Processing with Recurrent Networks Time will explain*.
https://web.stanford.edu/\~jurafsky/slp3/9.pdf
