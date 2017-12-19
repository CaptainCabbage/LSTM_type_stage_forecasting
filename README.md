Train and predict type and stage classes with LSTM using 8 features

## Folder structure
* ```type_classifier.py``` classifies 7 failure types
* ```stage_classifier.py``` classifies 10 stages

## Performance matrix

### Type classification
Model | Train loss | Train accuracy | Valid loss | Valid accuracy | Test loss | Test accuracy | Description
------|------------|----------------|------------|----------------|-----------|---------------|------------
v1 | 0.6362 | 0.8433 | 0.5260 | 0.8683 | 0.667 | 0.828 | full sequence
v2 | 0.1303 | 0.9628 | 0.5250 | 0.9005 | 0.495 | 0.884 | sequence of 32 time frames
v3 | 0.4276 | 0.9280 | 1.8071 | 0.8138 | 1.895 | 0.752 | sequence of 32 time frames with Keras stateful enabled
v4 | 0.2205 | 0.9334 | 0.3157 | 0.9211 | 0.381 | **0.896** | sequence of 32 time frames with dropout rate 0.2 at input layer

### Stage classification
Model | Train loss | Train accuracy | Valid loss | Valid accuracy | Test loss | Test accuracy | Description
------|------------|----------------|------------|----------------|-----------|---------------|------------
v1 | 0.5631 | 0.8630 | 0.4934 | 0.8817 | 0.596 | 0.844 | full sequence
v2 | 0.0882 | 0.9715 | 0.3694 | 0.9311 | 0.458 | 0.917 | sequence of 32 time frames
v3 | n/a | n/a | n/a | n/a | 0.864 | 0.851 | sequence of 32 time frames with Keras stateful enabled
v4 | 0.1290 | 0.9571 | 0.2943 | 0.9293 | 0.335 | **0.918** | sequence of 32 time frames with dropout rate 0.2 at input layer


## How to use

**IMPORTANT** The length of the sequence must be 32

### Infer the failure mode (type) classes for a single observation

```python
# Specify model and weights path
model_filename = os.path.join(os.path.dirname(__file__), '../models/type_lstm_v2.json')
weights_filename = os.path.join(os.path.dirname(__file__), '../models/type_lstm_v2.h5')

# Load the model
classifier = TypeClassifier(model_filename, weights_filename)

# Generate one observation with 32 sequences and 8 feature

num_timestamps = 32
num_features = 8
x = numpy.random.rand(num_timestamps, num_features)

# Inference
prediction = classifier.predict(x)

# The prediction is something like this
# 0 = {list} <type 'list'>: ['success', 'noscrew', 'no_hole_found', 'crossthread', 'stripped_no_engage', 'stripped', 'partial']
# 1 = {list} <type 'list'>: [[0.8852339386940002, 0.06836230307817459, 0.021053753793239594, 0.00986627209931612, 0.003368750214576721, 0.004927818197757006, 0.007187273818999529]]
```

### Infer the failure mode (type) classes for multiple observations
```python
# Specify model and weights path
model_filename = os.path.join(os.path.dirname(__file__), '../models/type_lstm_v2.json')
weights_filename = os.path.join(os.path.dirname(__file__), '../models/type_lstm_v2.h5')

# Load the model
classifier = TypeClassifier(model_filename, weights_filename)

# Generate 5 observations with 32 sequences and 8 feature

num_observations = 5
num_timestamps = 32
num_features = 8
x = numpy.random.rand(num_observations, num_timestamps, num_features)

# Inference
prediction = classifier.predict(x)

# The prediction is something like this
# 0 = {list} <type 'list'>: ['success', 'noscrew', 'no_hole_found', 'crossthread', 'stripped_no_engage', 'stripped', 'partial']
# 1 = {list} <type 'list'>: [[0.8852339386940002, 0.06836230307817459, 0.021053753793239594, 0.00986627209931612, 0.003368750214576721, 0.004927818197757006, 0.007187273818999529], ...]
```

### Infer the stage classes for a single observation

```python
# Specify model and weights path
model_filename = os.path.join(os.path.dirname(__file__), '../models/stage_lstm_v2.json')
weights_filename = os.path.join(os.path.dirname(__file__), '../models/stage_lstm_v2.h5')

# Load the model
classifier = TypeClassifier(model_filename, weights_filename)

# Generate one observation with 32 sequences and 8 feature

num_timestamps = 32
num_features = 8
x = numpy.random.rand(num_timestamps, num_features)

# Inference
prediction = classifier.predict(x)

# The prediction is something like this
# 0 = {list} <type 'list'>: ['approach', 'hole_finding', 'initial_mating', 'no_screw_spinning', 'rundown', 'screw_fallen', 'stripped_engaging', 'stripped_rundown', 'stripped_tightening', 'tightening']
# 1 = {list} <type 'list'>: [[0.8852339386940002, 0.06836230307817459, 0.021053753793239594, 0.00986627209931612, 0.003368750214576721, 0.004927818197757006, 0.007187273818999529, ...]]
```

### Infer the stage classes for multiple observations
```python
# Specify model and weights path
model_filename = os.path.join(os.path.dirname(__file__), '../models/stage_lstm_v2.json')
weights_filename = os.path.join(os.path.dirname(__file__), '../models/stage_lstm_v2.h5')

# Load the model
classifier = TypeClassifier(model_filename, weights_filename)

# Generate 5 observations with 32 sequences and 8 feature

num_observations = 5
num_timestamps = 32
num_features = 8
x = numpy.random.rand(num_observations, num_timestamps, num_features)

# Inference
prediction = classifier.predict(x)

# The prediction is something like this
# 0 = {list} <type 'list'>: ['approach', 'hole_finding', 'initial_mating', 'no_screw_spinning', 'rundown', 'screw_fallen', 'stripped_engaging', 'stripped_rundown', 'stripped_tightening', 'tightening']
# 1 = {list} <type 'list'>: [[0.8852339386940002, 0.06836230307817459, 0.021053753793239594, 0.00986627209931612, 0.003368750214576721, 0.004927818197757006, 0.007187273818999529, ...], ...]
```
