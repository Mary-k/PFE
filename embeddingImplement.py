
import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout
from keras.models import Sequential



my_df = pd.read_csv('/home/mimi/Desktop/PFE/datasets.csv',header=None)

x=my_df[0].values.tolist()
y=my_df[1].values.tolist()

from sklearn.model_selection import train_test_split

SEED = 2000
x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(x, y, test_size=.02, random_state=SEED) #

x_validation, x_test, y_validation, y_test = train_test_split(x_validation_and_test, y_validation_and_test, test_size=.5, random_state=SEED)



from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import TaggedDocument
import multiprocessing
from sklearn import utils


def labelize_tweets_ug(tweets,label):
    result = []
    prefix = label

    for i, t in zip (tweets.index, tweets):
        result.append(TaggedDocument(t.split(), [prefix + '_%s' % i]))
    return result

x_train_df=pd.DataFrame(x_train)
x_validation_df=pd.DataFrame(x_validation)
x_test_df=pd.DataFrame(x_test)

all_x = pd.concat([x_train_df,x_validation_df,x_test_df])
#all_x_list=all_x[0].values.tolist()

all_x_w2v = labelize_tweets_ug(all_x[0], 'all')

cores = multiprocessing.cpu_count()
model_ug_cbow = Word2Vec(sg=0, size=100, negative=5, window=2, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
model_ug_cbow.build_vocab([x.words for x in tqdm(all_x_w2v)])

for epoch in range(30):
    model_ug_cbow.train(utils.shuffle([x.words for x in tqdm(all_x_w2v)]), total_examples=len(all_x_w2v), epochs=1)
    model_ug_cbow.alpha -= 0.002
    model_ug_cbow.min_alpha = model_ug_cbow.alpha

model_ug_sg = Word2Vec(sg=1, size=100, negative=5, window=2, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
model_ug_sg.build_vocab([x.words for x in tqdm(all_x_w2v)])



for epoch in range(30):
    model_ug_sg.train(utils.shuffle([x.words for x in tqdm(all_x_w2v)]), total_examples=len(all_x_w2v), epochs=1)
    model_ug_sg.alpha -= 0.002
    model_ug_sg.min_alpha = model_ug_sg.alpha


model_ug_cbow.save('w2v_model_ug_cbow.word2vec')
model_ug_sg.save('w2v_model_ug_sg.word2vec')

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
'''
#
#
Separately trained Word2Vec (Average/Sum)
#
#
'''
def get_w2v_mean(tweet, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tweet.split():
        try:
            vec += np.append(model_ug_cbow[word],model_ug_sg[word]).reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec
  
def get_w2v_sum(tweet, size):
    vec = np.zeros(size).reshape((1, size))
    for word in tweet.split():
        try:
            vec += np.append(model_ug_cbow[word],model_ug_sg[word]).reshape((1, size))
        except KeyError:
            continue
    return vec
  
train_vecs_cbowsg_mean = scale(np.concatenate([get_w2v_mean(z, 200) for z in x_train]))
validation_vecs_cbowsg_mean = scale(np.concatenate([get_w2v_mean(z, 200) for z in x_validation]))
clf = LogisticRegression()
clf.fit(train_vecs_cbowsg_mean, y_train)
print (clf.score(validation_vecs_cbowsg_mean, y_validation))

train_vecs_cbowsg_sum = scale(np.concatenate([get_w2v_sum(z, 200) for z in x_train]))
validation_vecs_cbowsg_sum = scale(np.concatenate([get_w2v_sum(z, 200) for z in x_validation]))
clf = LogisticRegression()
clf.fit(train_vecs_cbowsg_sum, y_train)
print ('result ********* ',clf.score(validation_vecs_cbowsg_sum, y_validation))
'''
#
#
Separately trained Word2Vec with custom weighting (Average/Sum)
#
#
'''

w2v_pos_hmean_01 = {}
for w in model_ug_cbow.wv.vocab.keys():
    if w in pos_hmean.keys():
        w2v_pos_hmean_01[w] = np.append(model_ug_cbow[w],model_ug_sg[w]) * pos_hmean[w]
        
train_vecs_w2v_poshmean_mean_01 = scale(np.concatenate([get_w2v_general(z, 200, w2v_pos_hmean_01, 'mean') for z in x_train]))
validation_vecs_w2v_poshmean_mean_01 = scale(np.concatenate([get_w2v_general(z, 200, w2v_pos_hmean_01, 'mean') for z in x_validation]))
clf = LogisticRegression()
clf.fit(train_vecs_w2v_poshmean_mean_01, y_train)
print (clf.score(validation_vecs_w2v_poshmean_mean_01, y_validation))
train_vecs_w2v_poshmean_sum_01 = scale(np.concatenate([get_w2v_general(z, 200, w2v_pos_hmean_01, 'sum') for z in x_train]))
validation_vecs_w2v_poshmean_sum_01 = scale(np.concatenate([get_w2v_general(z, 200, w2v_pos_hmean_01, 'sum') for z in x_validation]))
clf = LogisticRegression()
clf.fit(train_vecs_w2v_poshmean_sum_01, y_train)
print (clf.score(validation_vecs_w2v_poshmean_sum_01, y_validation))

'''
#
#
Neural Network with Word2Vec
#
#
'''
train_w2v_final = train_vecs_w2v_poshmean_mean_01
validation_w2v_final = validation_vecs_w2v_poshmean_mean_01

from keras.callbacks import ModelCheckpoint, EarlyStopping

filepath="w2v_01_best_weights.{epoch:02d}-{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor='val_acc', patience=5, mode='max') 
callbacks_list = [checkpoint, early_stop]
np.random.seed(seed)
model_w2v_01 = Sequential()
model_w2v_01.add(Dense(256, activation='relu', input_dim=200))
model_w2v_01.add(Dense(256, activation='relu'))
model_w2v_01.add(Dense(256, activation='relu'))
model_w2v_01.add(Dense(1, activation='sigmoid'))
model_w2v_01.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model_w2v_01.fit(train_w2v_final, y_train, validation_data=(validation_w2v_final, y_validation), 
                 epochs=100, batch_size=32, verbose=2, callbacks=callbacks_list)
            
from keras.models import load_model
loaded_w2v_model = load_model('w2v_01_best_weights.10-0.8048.hdf5')
loaded_w2v_model.evaluate(x=validation_w2v_final, y=y_validation)