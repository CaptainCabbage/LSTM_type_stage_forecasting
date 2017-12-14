Train and predict type and stage classes with LSTM using 8 features

* Load data from CSV files
    * inspect the data
* Dependent variables
    * convert a class vector (integers) to binary class matrix (one hot encoding)
* Train a LSTM model for type classification
    * save and restore model and weights
    * evaluation and prediction
* Train a LSTM model for stage classification
    * save and restore model and weights
    * evaluation and prediction

Because Keras requries sequences of the same length in a batch, we pad zeros at the end of all sequences to match the longest sequence.
Each sequence is a data column.
