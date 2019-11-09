########################################################
# CLASSIFY TRANSACTIONS USING NAIVE BAYES CLASSIFIER
########################################################
import pickle
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import f1_score

with open('data/X_train_tfidf.pickle', 'rb') as data:
    X_train_tfidf = pickle.load(data)

with open('data/Y_train.pickle', 'rb') as data:
    Y_train = pickle.load(data)

with open('data/X_test_tfidf.pickle', 'rb') as data:
    X_test_tfidf = pickle.load(data)

with open('data/Y_test.pickle', 'rb') as data:
    Y_test = pickle.load(data)

naive_bayes_clf = ComplementNB()
naive_bayes_clf.fit(X_train_tfidf, Y_train)
naive_bayes_predictions = naive_bayes_clf.predict(X_test_tfidf)

# F1 on training dataset (in order to check for overfitting)
print(f1_score(Y_train, naive_bayes_clf.predict(X_train_tfidf), average='weighted'))

# F1 on test dataset
print(f1_score(Y_test, naive_bayes_predictions, average='weighted'))