import pickle
import pandas as pd
import numpy as np

from feature_engineering_util import create_transaction_type_feature, preprocess_transaction_description
from result_util import get_classification_results


def make_predictions_with_naive_bayes(model_path: str, data: pd.DataFrame):
    with open(model_path, 'rb') as file_content:
        naive_bayes_clf = pickle.load(file_content)

    naive_bayes_predictions_proba = naive_bayes_clf.predict_proba(data)

    return [naive_bayes_predictions_proba, naive_bayes_clf.classes_]


def make_predictions_with_svc(model_path: str, data: pd.DataFrame):
    with open(model_path, 'rb') as file_content:
        svc_clf = pickle.load(file_content)

    svc_predictions_proba = svc_clf.predict_proba(data)

    return [svc_predictions_proba, svc_clf.classes_]


# Load the data
with open('models/tfidf_vectorizer.pickle', 'rb') as content:
    tfidf_vectorizer = pickle.load(content)

with open('data/X_train_tfidf.pickle', 'rb') as content:
    X_train_tfidf = pickle.load(content)

with open('data/Y_train.pickle', 'rb') as content:
    Y_train = pickle.load(content)

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
predictions_proba, classes = make_predictions_with_naive_bayes("models/naive_bayes.pickle",
                                                               scorecard_data1_tfidf)

predictions_proba_df = pd.DataFrame(predictions_proba,
                                    columns=classes)


# Next, we will make the predictions based on the class probabilities, but with taking into an account the constraints
# about the transaction_type value specified in the transaction_class_to_type_dict dictionary.
transaction_class_to_type_dict = {
    "payment": "credit",
    "bank-fee": "credit",
    "cash-withdrawal": "credit",
    "interest": "debit",
    "refund": "debit",
    "transfer": "both"
}

# add two empty columns 'transaction_class' and 'confidence'
scorecard_data = pd.concat([scorecard_data, pd.DataFrame(columns=["transaction_class", "confidence"])], sort=False)

number_of_wrong_label_account_type = 0

for index, row in predictions_proba_df.iterrows():
    sorted_row = row.sort_values(ascending=False)

    # read the 'transaction_type' value assigned based on the 'transaction_amount'
    assigned_transaction_type = scorecard_data1.loc[index, "transaction_type"]

    for label, prob in sorted_row.items():
        if (assigned_transaction_type == transaction_class_to_type_dict[label])\
                | (transaction_class_to_type_dict[label] == "both"):

            # set the values in the original scorecard_data
            scorecard_data.at[index, ["transaction_class", "confidence"]] = [label, prob]
            break
        else:
            number_of_wrong_label_account_type += 1

print("Class label has been corrected", number_of_wrong_label_account_type,\
      "times taking into account the actual account_type value (credit/debit)")


scorecard_data.to_csv("data/scorecard_final.csv", index=False)
