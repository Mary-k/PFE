from sklearn import metrics, svm
from sklearn.model_selection import train_test_split
import pandas as pd  
import numpy as np
import gensim
import nltk
from gensim import corpora, models, similarities
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.models import KeyedVectors
from sklearn.preprocessing import scale


import time
from sklearn.metrics import classification_report

model_ug_cbow = KeyedVectors.load('w2v_model_ug_cbow.word2vec')
model_ug_sg = KeyedVectors.load('w2v_model_ug_sg.word2vec')


print(model_ug_sg.most_similar('رياض'))


'''
train_vecs_cbowsg_mean = scale(np.concatenate([em.get_w2v_mean(z, 200) for z in em.x_train]))
validation_vecs_cbowsg_mean = scale(np.concatenate([em.get_w2v_mean(z, 200) for z in em.x_validation]))
print(em.x_train)


classifier_linear = svm.SVC(kernel='linear')
t0 = time.time()


classifier_linear.fit(train_vecs_cbowsg_mean, em.y_train)
accuracy=classifier_linear.score(train_vecs_cbowsg_mean, em.y_train)
print(accuracy)
'''