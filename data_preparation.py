import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

from BasiqTfidfVectorizer import BasiqTfidfVectorizer
from feature_engineering_utils import preprocess_transaction_description, create_transaction_type_feature, \
    concat_tfidf_vector_with_factor_features

# load the transactions dataset
transactions = pd.read_csv("data/train-dataset.csv")


####################
# INSPECT THE DATA
####################

# get the total number of observations
print(transactions.shape[0])
# 100000

# get column names
list(transactions.columns)
# ['transaction_description', 'transaction_amount', 'transaction_account_type', 'transaction_class']

# get column information
transactions.info()
# transaction_description     100000 non-null object
# transaction_amount          100000 non-null float64
# transaction_account_type    100000 non-null object
# transaction_class           100000 non-null object

# see some examples
print(transactions.head(5))

transactions.describe()
#        transaction_amount
# count        1.000000e+05
# mean        -9.433642e+01
# std          9.194133e+03
# min         -2.150000e+06
# 25%         -6.360000e+01
# 50%         -1.280500e+01
# 75%          1.500000e-01
# max          7.000000e+05

# distribution for the column transaction_account_type
transactions.transaction_account_type.value_counts()
# transaction     78239
# savings         15457
# credit-card      4277
# term-deposit      812
# mortgage          570
# loan              427
# investment        218

# distribution for the column transaction_class
transactions.transaction_class.value_counts()
# payment            37814
# transfer           32990
# cash-withdrawal     9670
# interest            8602
# refund              7933
# bank-fee            2991

# Conclusion: the dataset is imbalanced. Precision/Recall or F1 should be used for measuring performance.


# Perform feature transformation
transactions["description_cleaned"] = preprocess_transaction_description(transactions["transaction_description"])
transactions['transaction_type'] = create_transaction_type_feature(transactions['transaction_amount'])

# Serialize to file
transactions.to_csv("data/transactions_preprocessed.csv")

##############################
# CREATE TRAIN AND TEST SETS
##############################

X_train, X_test, Y_train, Y_test = train_test_split(
    transactions[["description_cleaned", "transaction_type", "transaction_account_type"]],
    transactions['transaction_class'],
    test_size=0.2,
    random_state=2)

# Serialize files
with open('data/X_train.pickle', 'wb') as output:
    pickle.dump(X_train, output)

with open('data/X_test.pickle', 'wb') as output:
    pickle.dump(X_test, output)

with open('data/Y_train.pickle', 'wb') as output:
    pickle.dump(Y_train, output)

with open('data/Y_test.pickle', 'wb') as output:
    pickle.dump(Y_test, output)

######################
# CALCULATE TF-IDF
######################

# create a TF-IDF vectorizer
tfidf_vectorizer = BasiqTfidfVectorizer()

# calculate the TF-IDF scores on the training dataset
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train["description_cleaned"])
print(X_train_tfidf.shape)


# Since tfidf_vector is an ndarray, we transform it to a data frame. Note that this new df has reset indices
# compared to the original df. Add dummy variables of transaction_type and transaction_account_type, but reset their
# indices in order to perform a successful concat.
X_train_tfidf = pd.concat([pd.DataFrame(X_train_tfidf, columns=tfidf_vectorizer.get_feature_names()),
                           pd.get_dummies(X_train['transaction_type'].reset_index(drop=True)),
                           pd.get_dummies(X_train['transaction_account_type'].reset_index(drop=True))],
                          axis=1)

# calculate the TF-IDF scores for the test dataset
X_test_tfidf = tfidf_vectorizer.transform(X_test["description_cleaned"])
print(X_test_tfidf.shape)

# similar to the previous, convert it to df
X_test_tfidf = tfidf_vectorizer.concat_tfidf_vector_with_factor_features(X_test_tfidf,
                                                                         X_test['transaction_type'],
                                                                         X_test['transaction_account_type'])

# Serialize files
with open('data/X_train_tfidf.pickle', 'wb') as output:
    pickle.dump(X_train_tfidf, output)

with open('data/X_test_tfidf.pickle', 'wb') as output:
    pickle.dump(X_test_tfidf, output)

with open('models/basiq_tfidf_vectorizer.pickle', 'wb') as output:
    pickle.dump(tfidf_vectorizer, output)
