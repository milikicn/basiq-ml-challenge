########################################################
# CLASSIFY TRANSACTIONS USING NAIVE BAYES CLASSIFIER
########################################################
import pickle
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import f1_score
from result_util import store_classification_results

# Load the data

with open('data/X_train_tfidf.pickle', 'rb') as content:
    X_train_tfidf = pickle.load(content)

with open('data/Y_train.pickle', 'rb') as content:
    Y_train = pickle.load(content)

with open('data/X_test_tfidf.pickle', 'rb') as content:
    X_test_tfidf = pickle.load(content)

with open('data/Y_test.pickle', 'rb') as content:
    Y_test = pickle.load(content)

naive_bayes_clf = ComplementNB()
naive_bayes_clf.fit(X_train_tfidf, Y_train)
naive_bayes_predictions = naive_bayes_clf.predict(X_test_tfidf)


# F1 on training dataset (in order to check for overfitting)
f1_train = f1_score(Y_train, naive_bayes_clf.predict(X_train_tfidf), average='weighted')
print(f1_train)
# 0.944752498906083

# F1 on test dataset
f1_test = f1_score(Y_test, naive_bayes_predictions, average='weighted')
print(f1_test)
# 0.9447458900795492

bayes_model_path = 'models/naive_bayes.pickle'

# write the scores to the results.pickle
store_classification_results("naive_bayes", f1_train, f1_test, bayes_model_path)

# serialize the model
with open(bayes_model_path, 'wb') as output:
    pickle.dump(naive_bayes_clf, output)
