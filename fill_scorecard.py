import pickle
import pandas as pd

from feature_engineering_utils import create_transaction_type_feature, preprocess_transaction_description
from utils import get_classification_results


def make_predictions_with_naive_bayes(model_path: str, data: pd.DataFrame):
    with open(model_path, 'rb') as file_content:
        naive_bayes_clf = pickle.load(file_content)

    naive_bayes_predictions = naive_bayes_clf.predict(data)
    naive_bayes_predictions_proba = naive_bayes_clf.predict_proba(data)

    return [naive_bayes_predictions, naive_bayes_predictions_proba]


def make_predictions_with_svc(model_path: str, data: pd.DataFrame):
    with open(model_path, 'rb') as file_content:
        svc_clf = pickle.load(file_content)

    svc_predictions = svc_clf.predict(data)
    svc_predictions_proba = svc_clf.predict_proba(data)

    return [svc_predictions, svc_predictions_proba]


# Load the data
with open('models/tfidf_vectorizer.pickle', 'rb') as content:
    tfidf_vectorizer = pickle.load(content)

with open('data/X_train_tfidf.pickle', 'rb') as content:
    X_train_tfidf = pickle.load(content)


# load the scorecard data and run the preprocessing steps
scorecard_data = pd.read_csv("data/scorecard.csv")
scorecard_data1 = scorecard_data.copy()

scorecard_data1["description_cleaned"] = preprocess_transaction_description(scorecard_data1["transaction_description"])
scorecard_data1['transaction_type'] = create_transaction_type_feature(scorecard_data1['transaction_amount'])

# use previously serialized tfidf_vectorizer
scorecard_data1_tfidf = tfidf_vectorizer.transform(scorecard_data1["description_cleaned"]).toarray()
print(scorecard_data1_tfidf.shape)

# Since scorecard_data1_tfidf is an ndarray, we transform it to a data frame. Note that this new df has reset indices
# compared to the original df. Add dummy variables of transaction_type and transaction_account_type, but reset their
# indices in order to perform a successful concat.
scorecard_data1_tfidf = pd.concat([pd.DataFrame(scorecard_data1_tfidf, columns=tfidf_vectorizer.get_feature_names()),
                                   pd.get_dummies(scorecard_data1['transaction_type'].reset_index(drop=True),
                                                  prefix="type"),
                                   pd.get_dummies(scorecard_data1['transaction_account_type'].reset_index(drop=True),
                                                  prefix="account_type")],
                                  axis=1)

# since there can be more factor values in variables transaction_type and transaction_account_type, we will drop all
# columns that the original training set does not have
for column in scorecard_data1_tfidf.columns:
    if column not in X_train_tfidf.columns:
        scorecard_data1_tfidf.drop(column, axis=1, inplace=True)


# load the classification results file
results = get_classification_results()
print(results)

# From the results, we can conclude that CSV classifier has overfitted (the F1 score over the train set is too high).
# We can either try to remove some features from the dataset (reduce number of ngrams included in the model) or try
# reducing the C parameter in order to reduce the misclassification penalty.

# Since the SVC classifier is overfitting, we will use the next best classifier Naive Bayes.

# Load the serialized model of the best performing classifier, make predictions over the scorecard data
# and fill in the scorecard
predictions, proba = make_predictions_with_naive_bayes("models/naive_bayes.pickle",
                                                       scorecard_data1_tfidf)

scorecard_data['transaction_class'] = predictions
scorecard_data['confidence'] = [max(prob) for prob in proba]

scorecard_data.to_csv("data/scorecard_final.csv", index=False)
