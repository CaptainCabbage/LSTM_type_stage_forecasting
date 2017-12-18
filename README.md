Train and predict type and stage classes with LSTM using 8 features

## Folder structure
* ```type_classifier.py``` classifies 7 failure types
* ```stage_classifier.py``` classifies 9 stage types
    * ~~There are actually 10 stage types. I'll fix the model.~~ Fixed in models version 2 (i.e. *v2.*)

## How to use

**IMPORTANT The length of the sequence must be 50**

### Infer the failure modes(type) class for a single observation

```python
# Specify model and weights path
model_filename = os.path.join(os.path.dirname(__file__), '../models/type_lstm_v2.json')
weights_filename = os.path.join(os.path.dirname(__file__), '../models/type_lstm_v2.h5')

# Load the model
classifier = TypeClassifier(model_filename, weights_filename)

# Generate one observation with 50 sequences and 8 feature

num_timestamps = 50
num_features = 8
x = numpy.random.rand(num_timestamps, num_features)

# Inference
prediction = classifier.predict(x)

# The prediction is something like this
# 0 = {list} <type 'list'>: ['success', 'noscrew', 'no_hole_found', 'crossthread', 'stripped_no_engage', 'stripped', 'partial']
# 1 = {list} <type 'list'>: [[0.8852339386940002, 0.06836230307817459, 0.021053753793239594, 0.00986627209931612, 0.003368750214576721, 0.004927818197757006, 0.007187273818999529]]
```

### Infer the failure modes(type) class for multiple observations
```python
# Specify model and weights path
model_filename = os.path.join(os.path.dirname(__file__), '../models/type_lstm_v2.json')
weights_filename = os.path.join(os.path.dirname(__file__), '../models/type_lstm_v2.h5')

# Load the model
classifier = TypeClassifier(model_filename, weights_filename)

# Generate 5 observations with 50 sequences and 8 feature

num_observations = 5
num_timestamps = 50
num_features = 8
x = numpy.random.rand(num_observations, num_timestamps, num_features)

# Inference
prediction = classifier.predict(x)

# The prediction is something like this
# 0 = {list} <type 'list'>: ['success', 'noscrew', 'no_hole_found', 'crossthread', 'stripped_no_engage', 'stripped', 'partial']
# 1 = {list} <type 'list'>: [[0.8852339386940002, 0.06836230307817459, 0.021053753793239594, 0.00986627209931612, 0.003368750214576721, 0.004927818197757006, 0.007187273818999529], ...]
```
