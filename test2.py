from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


my_df = pd.read_csv('/home/mimi/Desktop/PFE/DATASETS/datasets.csv',header=None)
x=my_df[0].values.tolist()
y=my_df[1].values.tolist()


SEED = 100
x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(x, y, test_size=.02, random_state=SEED)
x_validation, x_test, y_validation, y_test = train_test_split(x_validation_and_test, y_validation_and_test, test_size=.5, random_state=SEED)

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
y_validation=replaced(y_validation)
y_test=replaced(y_test)


model_ug_cbow = KeyedVectors.load('w2v_model_ug_cbow.word2vec')
model_ug_sg = KeyedVectors.load('w2v_model_ug_sg.word2vec')
embeddings_index = {}
for w in model_ug_cbow.wv.vocab.keys():
    embeddings_index[w] = np.append(model_ug_cbow.wv[w],model_ug_sg.wv[w])

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=50)
tokenizer.fit_on_texts(x_train)
sequences = tokenizer.texts_to_sequences(x_train)

#print(sequences[:5])
'''
for x in sequences[:5]:
    print (x)
'''
length = []
for x in x_train:
    length.append(len(x.split()))

print(max(length))

max_length=1219+5

x_train_seq = pad_sequences(sequences, maxlen=max_length)
print(x_train_seq[:5])

sequences_val = tokenizer.texts_to_sequences(x_validation)
x_val_seq = pad_sequences(sequences_val, maxlen=max_length)


num_words = 50
embedding_matrix = np.zeros((num_words, 200))
for word, i in tokenizer.word_index.items():
    if i >= num_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

#from sklearn import metrics, svm



from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers.embeddings import Embedding


'''
model_ptw2v = Sequential()
e = Embedding(50, 200, weights=[embedding_matrix], input_length=max_length, trainable=False)
model_ptw2v.add(e)
model_ptw2v.add(Flatten())
model_ptw2v.add(Dense(256, activation='relu'))
model_ptw2v.add(Dense(1, activation='sigmoid'))
model_ptw2v.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_ptw2v.fit(x_train_seq, y_train, validation_data=(x_val_seq, y_validation), epochs=5, batch_size=15, verbose=2)
'''
from keras.layers import Conv1D, GlobalMaxPooling1D
structure_test = Sequential()
e = Embedding(50, 200, input_length=45)
structure_test.add(e)
structure_test.add(Conv1D(filters=100, kernel_size=2, padding='valid', activation='relu', strides=1))
structure_test.summary()

structure_test = Sequential()
e = Embedding(50, 200, input_length=45)
structure_test.add(e)
structure_test.add(Conv1D(filters=100, kernel_size=2, padding='valid', activation='relu', strides=1))
structure_test.add(GlobalMaxPooling1D())
structure_test.summary()

model_cnn_01 = Sequential()
e = Embedding(50, 200, weights=[embedding_matrix], input_length=45, trainable=False)
model_cnn_01.add(e)
model_cnn_01.add(Conv1D(filters=100, kernel_size=2, padding='valid', activation='relu', strides=1))
model_cnn_01.add(GlobalMaxPooling1D())
model_cnn_01.add(Dense(256, activation='relu'))
model_cnn_01.add(Dense(1, activation='sigmoid'))
model_cnn_01.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_cnn_01.fit(x_train_seq, y_train, validation_data=(x_val_seq, y_validation), epochs=5, batch_size=32, verbose=2)
