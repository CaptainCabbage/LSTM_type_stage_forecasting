Train and predict type and stage classes with LSTM using 8 features

## Folder structure
* ```type_classifier.py``` classifies 7 failure types
* ```stage_classifier.py``` classifies 9 stage types
    * **Note** There are actually 10 stage types. I'll fix the model.

## How to use
```python
# Specify model and weights path
model_filename = os.path.join(os.path.dirname(__file__), '../models/type_lstm.json')
weights_filename = os.path.join(os.path.dirname(__file__), '../models/type_lstm.h5')

# Load the model
classifier = TypeClassifier(model_filename, weights_filename)

# Generate one observation with 8 feature and of 4479 length
nums_timestamp = 4479
num_features = 8
x = numpy.random.rand(nums_timestamp, num_features)

# Inference
prediction = classifier.predict(x)

# The prediction is something like this
# 0 = {list} <type 'list'>: ['success', 'noscrew', 'no_hole_found', 'crossthread', 'stripped_no_engage', 'stripped', 'partial']
# 1 = {list} <type 'list'>: [[0.8852339386940002, 0.06836230307817459, 0.021053753793239594, 0.00986627209931612, 0.003368750214576721, 0.004927818197757006, 0.007187273818999529]]
```

## Training notes
Because Keras requires sequences of the same length in a batch, we pad zeros at the end of all sequences to match the longest sequence.
