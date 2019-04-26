import embeddingImplement as em
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
import numpy as np


train_vecs_cbowsg_mean = scale(np.concatenate([em.get_w2v_mean(z, 200) for z in em.x_train]))
validation_vecs_cbowsg_mean = scale(np.concatenate([em.get_w2v_mean(z, 200) for z in em.x_validation]))
print(train_vecs_cbowsg_mean)


clf = LogisticRegression()

clf.fit(train_vecs_cbowsg_mean, em.y_train)
print (clf.score(validation_vecs_cbowsg_mean,em.y_validation))

train_vecs_cbowsg_sum = scale(np.concatenate([em.get_w2v_sum(z, 200) for z in em.x_train]))
validation_vecs_cbowsg_sum = scale(np.concatenate([em.get_w2v_sum(z, 200) for z in em.x_validation]))
clf = LogisticRegression()
clf.fit(train_vecs_cbowsg_sum, em.y_train)
print (clf.score(validation_vecs_cbowsg_sum, em.y_validation))