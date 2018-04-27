# Keras Attention
With this project, you can easily use the GRU attention model on Keras.

# Example

```python
from keras.layers import Input, Dense, Embedding
from keras.models import Model
from .models import AttentionGRU

N_MAX_WORDS = 100
N_DICTOINARY = 10000
N_EMBED_DIM = 256

x = Input((N_MAX_WORDS,))
e = Embedding(output_dim=N_EMBED_DIM,
              input_dim=N_DICTOINARY,
              input_length=N_MAX_WORDS,
              trainable=True)(x)
g = AttentionGRU(input_shape=(None, N_MAX_WORDS, N_EMBED_DIM))(e)
o = Dense(self.nb_issues, activation='softmax')(g)
model = Model(inputs=x, outputs=o)
```