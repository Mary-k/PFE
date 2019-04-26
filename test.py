import pandas as pd  
import numpy as np
import gensim
import nltk
from gensim import corpora, models, similarities
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.models import KeyedVectors


model_ug_cbow = KeyedVectors.load('w2v_model_ug_cbow.word2vec')
model_ug_sg = KeyedVectors.load('w2v_model_ug_sg.word2vec')


print(y_train[20])
def replaced(sequence):
    new=[]

    for target in sequence:
        if target == ' Negative':
            target=0
            new.append(0)
        elif target == ' Positive':
            new.append(1)
        else:
            new.append(2)
    return new


y_train=replaced(y_train)


print(y_train[20])



x_validation=validation[0].values.tolist()
y_validation=validation[1].values.tolist()

y_validation=replaced(y_validation)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences




embeddings_index = {}

for w in model_ug_cbow.wv.vocab.keys():  
    embeddings_index[w] = np.append(model_ug_cbow.wv[w],model_ug_sg.wv[w])    


#print(x_train)

tokenizer = Tokenizer(num_words=100000)
tokenizer.fit_on_texts(x_train)
sequences = tokenizer.texts_to_sequences(x_train)

#print(sequences[:5])
length = []
for z in x_train:
    length.append(len(z.split(' ')))

#print(max(length))
x_train_seq = pad_sequences(sequences, maxlen=1221)
#print(x_train_seq[:5])

tokenizer.fit_on_texts(x_validation)
sequences_val = tokenizer.texts_to_sequences(x_validation)
x_val_seq = pad_sequences(sequences_val, maxlen=1221)

num_words = 100000
embedding_matrix = np.zeros((num_words, 200))
for word, i in tokenizer.word_index.items():
    if i >= num_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


seed = 7

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers.embeddings import Embedding




model_ptw2v = Sequential()
e = Embedding(100000, 200, weights=[embedding_matrix], input_length=1221, trainable=True)
model_ptw2v.add(e)
model_ptw2v.add(Flatten())
model_ptw2v.add(Dense(256, activation='relu'))
model_ptw2v.add(Dense(1, activation='sigmoid'))
model_ptw2v.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_ptw2v.fit(x_train_seq, y_train, validation_data=(x_val_seq, y_validation), epochs=5, batch_size=32, verbose=2)
