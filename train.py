from keras.datasets import imdb
from keras.layers import Input, Dense, Embedding
from keras.models import Model
from keras.preprocessing import sequence
from keras.utils import np_utils
from src.models import AttentionGRU


N_MAX_WORDS = 300
N_EMBED_DIM = 256
N_CLASSES = 2


def read_dataset():
    # Load IMDB Sentiment Dataset
    (x_train, y_train), (x_test, y_test) = imdb.load_data()
    
    # Create dictonary
    word_ids = set([word_id for x in x_train for word_id in x])
    
    # Preprocess
    x_train = sequence.pad_sequences(x_train, maxlen=N_MAX_WORDS)
    x_test = sequence.pad_sequences(x_test, maxlen=N_MAX_WORDS)
    x_test = [word if word in word_ids else 0 for words in x_test for word in words]
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    
    return word_ids, x_train, y_train, x_test, y_test
    

def main():
    # Read a dataset
    word_ids, x_train, y_train, x_test, y_test = read_dataset()
    
    # Generate a model
    x = Input((N_MAX_WORDS,))
    e = Embedding(output_dim=N_EMBED_DIM,
                  input_dim=len(word_ids),
                  input_length=N_MAX_WORDS,
                  trainable=True)(x)
    g = AttentionGRU(input_shape=(None, N_MAX_WORDS, N_EMBED_DIM))(e)
    o = Dense(N_CLASSES, activation='softmax')(g)
    model = Model(inputs=x, outputs=o)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.fit(x_train, y_train, batch_size=256, epochs=10)
    score = model.evaluate(x_test, y_test, batch_size=128)
    print(score)


if __name__ == '__main__':
    main()
