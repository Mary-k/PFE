import embeddingImplement as em
from gensim.models import KeyedVectors
from sklearn.preprocessing import scale
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report


train_vecs_cbowsg_mean = scale(np.concatenate([em.get_w2v_mean(z, 200) for z in em.x_train]))
validation_vecs_cbowsg_mean = scale(np.concatenate([em.get_w2v_mean(z, 200) for z in em.x_validation]))


forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(train_vecs_cbowsg_mean, em.y_train)

result = forest.predict(validation_vecs_cbowsg_mean)

print(classification_report(em.y_validation, result))

