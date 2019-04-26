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
import embeddingImplement as em

train_vecs_cbowsg_mean = scale(np.concatenate([em.get_w2v_mean(z, 200) for z in em.x_train]))
validation_vecs_cbowsg_mean = scale(np.concatenate([em.get_w2v_mean(z, 200) for z in em.x_validation]))

classifier_linear = svm.SVC(kernel='linear')
t0 = time.time()


classifier_linear.fit(train_vecs_cbowsg_mean, em.y_train)
accuracy=classifier_linear.score(train_vecs_cbowsg_mean, em.y_train)
print(accuracy)
