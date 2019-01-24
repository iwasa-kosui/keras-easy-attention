from keras import backend as K
from keras.layers import Input, Dense, Embedding, GRU, TimeDistributed, Concatenate, RepeatVector, Permute, Lambda, Bidirectional, Multiply
from keras.models import Model

def AttentionGRU(input_shape, gru_dim=256, dropout=0.5, return_sequences=False):
    batch_size, time_steps, embedding_dim = input_shape
    i = Input(shape=(time_steps, embedding_dim, ), dtype='float32')
    g = Bidirectional(GRU(gru_dim, dropout=dropout, return_sequences=True))(i)
    a = Permute((2, 1))(g)
    a = Dense(time_steps, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    m = Multiply(name='attention_mul')([g, a_probs])
    if return_sequences:
        return Model(i, m, name='attention')
    o = Lambda(lambda x: K.sum(x, axis=1))(m)
    return Model(i, o, name='attention')
