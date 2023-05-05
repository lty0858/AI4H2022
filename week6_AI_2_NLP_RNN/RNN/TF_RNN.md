#

- [LONG SHORT-TERM MEMORY(1997)](https://www.bioinf.jku.at/publications/older/2604.pdf)
- [GRU(2014) Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078)

## [Recurrent Neural Networks (RNN) with Keras](https://www.tensorflow.org/guide/keras/rnn)
- The Keras RNN API:  the built-in 
- [tf.keras.layers.RNN]()
- [tf.keras.layers.LSTM]() 
- [tf.keras.layers.GRU]()
```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
```

```python
model = keras.Sequential()
# Add an Embedding layer expecting input vocab of size 1000, and
# output embedding dimension of size 64.
model.add(layers.`Embedding`(input_dim=1000, output_dim=64))

# Add a LSTM layer with 128 internal units.
model.add(layers.`LSTM`(128))

# Add a Dense layer with 10 units.
model.add(layers.Dense(10))

model.summary()
```


```python
model = keras.Sequential()
model.add(layers.`Embedding`(input_dim=1000, output_dim=64))

# The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 256)
model.add(layers.`GRU`(256, return_sequences=True))

# The output of SimpleRNN will be a 2D tensor of shape (batch_size, 128)
model.add(layers.`SimpleRNN`(128))

model.add(layers.Dense(10))

model.summary()
```

