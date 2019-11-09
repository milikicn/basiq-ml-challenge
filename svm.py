#######################################
# CLASSIFY TRANSACTIONS USING SVM
#######################################
import pickle
from sklearn import svm
from sklearn.model_selection import RandomizedSearchCV, ShuffleSplit, GridSearchCV

with open('data/X_train_tfidf.pickle', 'rb') as data:
    X_train_tfidf = pickle.load(data)

with open('data/Y_train.pickle', 'rb') as data:
    Y_train = pickle.load(data)

with open('data/X_test_tfidf.pickle', 'rb') as data:
    X_test_tfidf = pickle.load(data)

with open('data/Y_test.pickle', 'rb') as data:
    Y_test = pickle.load(data)

# HYPERPARAMETER TUNING

# C: penalty parameter
# kernel: kernel to be used
# gamma: kernel coefficient
# degree: degree of the polynomial kernel function


# Perform random search of hyperparameters

C_range = [.0001, .001, .01]

# trying only with linear kernel because of the performance. Ideally, 'rbf', 'poly' should be included. In this case,
# we would first apply RandomizedSearchCV which randomly selects a combination of parameter values, performs the fit and
# calculate the value. But since we will only try with a 'linear' kernel, then we will jump to GridSearchCV.
kernel_range = ['linear']

# in case 'poly' kernel is used, then we would vary the 'degree' parameter, e.g. 1, 2, 3, 4, 5]
# in case 'rbf' kernel is used, then we would vary the 'gamma' parameter, e.g. .0001, .001, .01, .1, 1, 10, 100

# Create the random grid
param_grid = {'C': C_range,
              'gamma': gamma_range,
              'degree': degree_range,
              'kernel': kernel_range,
              'probability': [True]
              }

svc_clf = svm.SVC(random_state=123)

random_search = RandomizedSearchCV(estimator=svc_clf,
                                   param_distributions=param_grid,
                                   n_iter=10,  # this should be increased to e.g. 30 or 50
                                   scoring='f1_weighted',
                                   cv=3,  # this should be increased to e.g. 5 (new default value)
                                   verbose=1,
                                   random_state=123)

# fit the random search model
random_search.fit(X_train_tfidf, Y_train)

# the best hyperparameters are:
print(random_search.best_params_)

# /usr/local/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
#   'precision', 'predicted', average, warn_for)
# [Parallel(n_jobs=1)]: Done  30 out of  30 | elapsed: 652.0min finished
# Out[4]:
# RandomizedSearchCV(cv=3, error_score='raise-deprecating',
#                    estimator=SVC(C=1.0, cache_size=200, class_weight=None,
#                                  coef0=0.0, decision_function_shape='ovr',
#                                  degree=3, gamma='auto_deprecated',
#                                  kernel='rbf', max_iter=-1, probability=False,
#                                  random_state=123, shrinking=True, tol=0.001,
#                                  verbose=False),
#                    iid='warn', n_iter=10, n_jobs=None,
#                    param_distributions={'C': [0.0001, 0.001, 0.01],
#                                         'degree': [1, 2, 3, 4, 5],
#                                         'gamma': [0.0001, 0.001, 0.01, 0.1, 1,
#                                                   10, 100],
#                                         'kernel': ['linear', 'rbf', 'poly'],
#                                         'probability': [True]},
#                    pre_dispatch='2*n_jobs', random_state=123, refit=True,
#                    return_train_score=False, scoring='f1_weighted', verbose=1)
# print(random_search.best_params_)
# {'probability': True, 'kernel': 'poly', 'gamma': 1, 'degree': 3, 'C': 0.01}


# Perform grid (exhausting) search of hyperparameters

C = [.001, .01, .1]
degree = [2, 3, 4]
gamma = [.1, 1, 10]
probability = [True]

param_grid1 = [
    {'C': C, 'kernel': ['linear'], 'probability': probability},
    {'C': C, 'kernel': ['poly'], 'degree': degree, 'probability': probability},
    {'C': C, 'kernel': ['rbf'], 'gamma': gamma, 'probability': probability}
]

svc_clf1 = svm.SVC(random_state=123)

# Instantiate the grid search model
grid_search = GridSearchCV(estimator=svc_clf1,
                           param_grid=param_grid1,
                           scoring='f1_weighted',
                           cv=3,
                           verbose=1)

# Fit the grid search to the data
grid_search.fit(X_train_tfidf, Y_train)

print("The best hyperparameters from Grid Search are:")
print(grid_search.best_params_)

print("The mean F1 score of a model with these hyperparameters is:")
print(grid_search.best_score_)

