import embeddingImplement as em
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split



my_df = pd.read_csv('/home/mimi/Desktop/PFE/DATASETS/datasets.csv',header=None)
x=my_df[0].values.tolist()
y=my_df[1].values.tolist()


SEED = 100
x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(x, y, test_size=.02, random_state=SEED)
x_validation, x_test, y_validation, y_test = train_test_split(x_validation_and_test, y_validation_and_test, test_size=.5, random_state=SEED)
 

train_vecs_cbowsg_mean = scale(np.concatenate([em.get_w2v_mean(z, 200) for z in x_train]))
validation_vecs_cbowsg_mean = scale(np.concatenate([em.get_w2v_mean(z, 200) for z in x_validation]))
print(train_vecs_cbowsg_mean)


clf = LogisticRegression()

clf.fit(train_vecs_cbowsg_mean, y_train)
print (clf.score(validation_vecs_cbowsg_mean,y_validation))

train_vecs_cbowsg_sum = scale(np.concatenate([em.get_w2v_sum(z, 200) for z in x_train]))
validation_vecs_cbowsg_sum = scale(np.concatenate([em.get_w2v_sum(z, 200) for z in x_validation]))
clf = LogisticRegression()
clf.fit(train_vecs_cbowsg_sum, y_train)
print (clf.score(validation_vecs_cbowsg_sum, y_validation))