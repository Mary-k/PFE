from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier

text = [line.rstrip('\n') for line in open('train_100.txt')]
simp = [line.rstrip('\n') for line in open('test_100.txt')]
vectorizer = CountVectorizer()
# vectorizer = CountVectorizer()

vectorizer.fit(text)
train = vectorizer.transform(text)
test = vectorizer.transform(simp)
transformer = TfidfTransformer(smooth_idf=False)
train = transformer.fit_transform(train)
test = transformer.fit_transform(test)

feat_train = train.toarray()
feat_test = test.toarray()
lab_test = np.zeros(400)
i = 0
for row in feat_test:
    #print len(row)
    if(i<100):
        lab_test[i] = 0
    if(i >= 100 and i<200):
        lab_test[i] = 1
    if(i>= 200 and i<300):
        lab_test[i] = 2
    if(i>= 400):
        lab_test[i] = 3
    i=i+1

mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=20, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)
mlp.fit(feat_train, lab_test)
print("Training set score: %f" % mlp.score(feat_train, lab_test))
print("Test set score: %f" % mlp.score(feat_test, lab_test))

# fig, axes = plt.subplots(4, 4)
# # use global min / max to ensure all weights are shown on the same scale
# vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
# for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
#     ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=.5 * vmin,
#                vmax=.5 * vmax)
#     ax.set_xticks(())
#     ax.set_yticks(())
#
# plt.show()
