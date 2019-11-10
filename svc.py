#######################################
# CLASSIFY TRANSACTIONS USING SVM
#######################################
import pickle
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from utils import store_classification_results


# Load the data

with open('data/X_train_tfidf.pickle', 'rb') as data:
    X_train_tfidf = pickle.load(data)

with open('data/Y_train.pickle', 'rb') as data:
    Y_train = pickle.load(data)

with open('data/X_test_tfidf.pickle', 'rb') as data:
    X_test_tfidf = pickle.load(data)

with open('data/Y_test.pickle', 'rb') as data:
    Y_test = pickle.load(data)


# HYPERPARAMETER TUNING

"""Because of the time and computing limitations of my local machine, I will present a simplified approach to seaching
for optimal parameters for the SVM classifier.

In the real project, the parameter tuning phase would include two steps:
1.    Randomized search on hyper parameters (class sklearn.model_selection.RandomizedSearchCV). This class allows
      supplying a range of values for different parameters that are to be tuned. A model with different combinations
      of parameter values is fitted  and the performance measure (F1 in our case) is measured. Cross-validation is
      applied here. Not all combinations of parameter values is used, but only a random number of their combinations
      (defined in the 'n_iter' parameter). The fit method will output the combination of parameter values with the
      highest score for the performance measure (F1).
2.    Grid search (exhaustive) over specified parameter values (class sklearn.model_selection.GridSearchCV). From the
      previous step, we found a random guess of the best parameter values to use. In this step, we define another
      ranges of values for all parameters that are around the values from the previous step."""

"""Parameter that are to be tuned:
- C: penalty parameter
- kernel: kernel to be used ('linear', 'rbf', 'poly')
- gamma: kernel coefficient. It makes sense to tune this parameter in case 'rbf' kernel is used. Then we would vary the 
    values of this parameter, e.g. .0001, .001, .01, .1, 1, 10, 100
- degree: degree of the polynomial kernel function. It makes sense to tune this parameter in case 'poly' kernel is used.
    Then we would vary the values of this parameter, e.g. 1, 2, 3, 4, 5]"""

"""In my case, due to time and computation resource limitations, I will use only 'linear' kernel since it is the 
simplest one, the fastest one, and with the least parameters to tune. This kernel only requires setting the C
parameter."""

C_range = [.0001, .001, .01, .1, 1, 10]

# Create the random grid
param_grid = {'C': C_range,
             'kernel': ['linear'],
             'probability': [True]
             }

svc_grid_clf = svm.SVC(random_state=123)

# Instantiate the grid search model
grid_search = GridSearchCV(estimator=svc_grid_clf,
                          param_grid=param_grid,
                          scoring='f1_weighted',
                          cv=3,
                          verbose=1)

# Fit the grid search to the data
grid_search.fit(X_train_tfidf, Y_train)

print("The best hyperparameters from Grid Search are:")
print(grid_search.best_params_)
#

# use the best performing parameter configuration for out classification problem
svc_clf = grid_search.best_estimator_
svc_clf.fit(X_train_tfidf, Y_train)
svc_pred = svc_clf.predict(X_test_tfidf)

# F1 on training dataset (in order to check for overfitting)
f1_train = f1_score(Y_train, svc_clf.predict(X_train_tfidf), average='weighted')
print(f1_train)
#

# F1 on test dataset
f1_test = f1_score(Y_test, svc_clf, average='weighted')
print(f1_test)
#

# write the scores to the results.pickle
store_classification_results("SVC", f1_train, f1_test)


# serialize the model
with open('models/svc.pickle', 'wb') as output:
   pickle.dump(svc_clf, output)
