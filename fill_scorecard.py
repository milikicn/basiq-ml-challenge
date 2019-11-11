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
    pass


def make_predictions_with_neural_network(model_path: str, data: pd.DataFrame):
    with open(model_path, 'rb') as file_content:
        neural_network_clf = pickle.load(file_content)
    pass


def make_predictions(algorithm, model_path, data):
    model_to_function = {
        "naive_bayes": make_predictions_with_naive_bayes,
        "svc": make_predictions_with_svc,
        "neural_network": make_predictions_with_neural_network,
    }

    return model_to_function[algorithm](model_path, data)


# Load the data
with open('models/basiq_tfidf_vectorizer.pickle', 'rb') as data:
    tfidf_vectorizer = pickle.load(data)

with open('data/X_train_tfidf.pickle', 'rb') as data:
    X_train_tfidf = pickle.load(data)


# load the scorecard data and prepare it for classification
scorecard_data = pd.read_csv("data/scorecard.csv")
scorecard_data1 = scorecard_data.copy()

scorecard_data1["description_cleaned"] = preprocess_transaction_description(scorecard_data1["transaction_description"])
scorecard_data1['transaction_type'] = create_transaction_type_feature(scorecard_data1['transaction_amount'])

# use previously serialized tfidf_vectorizer
scorecard_data1_tfidf = tfidf_vectorizer.transform(scorecard_data1["description_cleaned"])
print(scorecard_data1_tfidf.shape)

# Since scorecard_data1_tfidf is an ndarray, we transform it to a data frame. Note that this new df has reset indices
# compared to the original df. Add dummy variables of transaction_type and transaction_account_type, but reset their
# indices in order to perform a successful concat.
scorecard_data1_tfidf = pd.concat([pd.DataFrame(scorecard_data1_tfidf, columns=tfidf_vectorizer.get_feature_names()),
                                   pd.get_dummies(scorecard_data1['transaction_type'].reset_index(drop=True)),
                                   pd.get_dummies(scorecard_data1['transaction_account_type'].reset_index(drop=True))],
                                  axis=1)

# since there can be more factor values in variables transaction_type and transaction_account_type, we will drop all
# columns that the original training set does not have
for column in scorecard_data1_tfidf.columns:
    if column not in X_train_tfidf.columns:
        scorecard_data1_tfidf.drop(column, axis=1, inplace=True)


# load the classification results file
results = get_classification_results()

# find the best performing algorithm
best_score_info = results.loc[results['f1_test'].idxmax()]

print("The best performing algorithm is: " + best_score_info["algorithm"])

# Load the serialized model of the best performing classifier, make predictions over the scorecard data
# and fill in the scorecard
predictions, proba = make_predictions(best_score_info["algorithm"],
                                      best_score_info["model_path"],
                                      scorecard_data1_tfidf)

scorecard_data['transaction_class'] = predictions
scorecard_data['confidence'] = [max(prob) for prob in proba]

scorecard_data.to_csv("data/scorecard_final.csv")








