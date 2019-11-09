import pickle
import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

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

#######################
# FEATURE ENGINEERING
#######################

# 1. Lowercase the text
transactions['transaction_description1'] = transactions['transaction_description'].str.lower()
transactions['transaction_description1'].head(5)


# 2. Introduce DATETOKEN and TIMETOKEN

# Create regex expressions for all date formats found in the dataset. Date can be prefixed with the string "Date" or "Date:".
expanded_date_regex = "(date(:)?(\\s)?)?\\d{1,2}\\s+(jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|jun(e)?|jul(y)?|aug(ust)?|sep(tember)?|oct(ober)?|nov(ember)?|dec(ember)?)\\s+\\d{4}"
short_date_regex = "(date(:)?(\s)?)?[\d+]{1,2}[- /.][\d+]{1,2}[- /.][\d+]{4}"

# Count how many date occurrences there are
transactions['transaction_description1'].str.count(expanded_date_regex).sum()
# 9370
transactions['transaction_description1'].str.count(short_date_regex).sum()
transactions['transaction_description1'][89]
# 2020

# Replace both date format occurrences (along with prefixes if they are present) with the DATETOKEN
transactions['transaction_description2'] = transactions['transaction_description1'].str.replace(expanded_date_regex, "DATETOKEN")
transactions['transaction_description2'] = transactions['transaction_description2'].str.replace(short_date_regex, "DATETOKEN")

# Create a regex expression for the time format found in the dataset
time_regex = "(time(:)?\\s)?([0-9]{2}:[0-9]{2})(am|pm)?"

# Count how many time occurrences there are
transactions['transaction_description2'].str.count(time_regex).sum()
# 7522

# Replace time occurrences (along with prefix if it is present) with the TIMETOKEN
transactions['transaction_description2'] = transactions['transaction_description2'].str.replace(time_regex, "TIMETOKEN")


# 3. replace finance specific abbreviations with the expanded word. This should be expanded with more terms.
abbreviations_dict = {
  "w/d": "withdrawal"
}
transactions['transaction_description3'] = transactions['transaction_description2'].replace(abbreviations_dict, regex=True)


# 4. remove punctuation symbols
transactions['transaction_description4'] = transactions['transaction_description3'].apply(lambda x: x.translate(str.maketrans(string.punctuation,' '*32)))


# 5. Replace receipt number with RECEIPTTOKEN

# Create a regex expression that searches for e.g. "Receipt 167448"
receipt_number_regex = "(receipt)\s+([\w|*]{5,30})\\b"

# Count how many receipt number there are
transactions['transaction_description4'].str.count(receipt_number_regex).sum()
# 16843

# Replace receipt occurrences with the RECEIPTTOKEN
transactions['transaction_description5'] = transactions['transaction_description4'].str.replace(receipt_number_regex, "RECEIPTTOKEN")


# 6. Replace long numbers with a NUMBERTOKEN (that could be an indicator of an account number or some id).
# This is to be checked whether it makes sense

# Create a regex expression that searches for numbers with more that 4 digits. Also, support masked digits
# (i.e. 498691xxxxxx8454) or having any other letter (e.g. 709926s77.2)
long_number_regex = "[0-9]{4,15}(\w+)?([0-9]{1,10})"

# Count how many long number there are
transactions['transaction_description5'].str.count(long_number_regex).sum()
# 80328

# Replace long number occurrences with the NUMBERTOKEN
transactions['transaction_description6'] = transactions['transaction_description5'].str.replace(long_number_regex, "NUMBERTOKEN")


# 7. Perform lematization

# Downloading wordnet from NLTK
nltk.download('wordnet')

# Saving the whitespace tokenizer and WordNet lemmatizer
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

transactions['transaction_description7'] = transactions['transaction_description6'].apply(lambda x: " ".join([lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(x)]))


# 8. Remove stopwords

# Download the stop words list
nltk.download('stopwords')

# Load the stop words in english
stop_words = list(stopwords.words('english'))

# remove stopwords
transactions['transaction_description8'] = transactions['transaction_description7'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))


# 9. Remove stand-alone numbers
transactions['transaction_description9'] = transactions['transaction_description8'].str.replace("\\d+ ", " ")


# 10. Remove words with one or two letters (assuming they don't bring any value to the classifier)
transactions['transaction_description10'] = transactions['transaction_description9'].str.replace("\\b[\\w]{1,2}\\b", " ")


# 11. Trim whitespaces
transactions['transaction_description11'] = transactions['transaction_description10'].str.replace("\\s+", " ")


# Based on the transaction_amount value, create a new variable transaction_type
transactions['transaction_type'] = np.where(transactions['transaction_amount'] > 0, 'debit', 'credit')

# Serialize to file
with open('data/transactions.pickle', 'wb') as output:
    pickle.dump(transactions, output)

##############################
# CREATE TRAIN AND TEST SETS
##############################

X_train, X_test, Y_train, Y_test = train_test_split(
    transactions[["transaction_description10", "transaction_type", "transaction_account_type"]],
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

# configure the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(encoding='utf-8',
                                   ngram_range=(1,3),   # we want to create unigrams and bigrams
                                   stop_words=None,    # already applied
                                   lowercase=False,    # already applied
                                   max_df=0.95,  # remove all terms that have document frequency lower than 5th percentile
                                   min_df=0.05,  # remove all terms that have document frequency higher than 95th percentile
                                   max_features=500,
                                   norm='l2',
                                   sublinear_tf=True)  # apply sublinear tf scaling, replace tf with 1 + log(tf)

# calculate the TF-IDF scores for the training dataset
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train["transaction_description10"]).toarray()
print(X_train_tfidf.shape)

tfidf_vectorizer.get_feature_names()

# Create a training DF with tfidf values. Since X_train_tfidf is an ndarray, we transform it to a df. Note that
# this new df has reset indices compared to the original df. Add encoded variables transaction_type and
# transaction_account_type. Reset their indices in order to perform a successful concat.
X_train_tfidf = pd.concat([pd.DataFrame(X_train_tfidf, columns=tfidf_vectorizer.get_feature_names()),
                           pd.get_dummies(X_train['transaction_type'].reset_index(drop=True)),
                           pd.get_dummies(X_train['transaction_account_type'].reset_index(drop=True))],
                          axis=1)
print(X_train_tfidf.shape[0])

# calculate the TF-IDF scores for the test dataset
X_test_tfidf = tfidf_vectorizer.transform(X_test["transaction_description10"]).toarray()
print(X_test_tfidf.shape)

# similar to the previous, convert it to df
X_test_tfidf = pd.concat([pd.DataFrame(X_test_tfidf, columns=tfidf_vectorizer.get_feature_names()),
                          pd.get_dummies(X_test['transaction_type'].reset_index(drop=True)),
                          pd.get_dummies(X_test['transaction_account_type'].reset_index(drop=True))],
                         axis=1)
print(X_test_tfidf.shape[0])


# Serialize files
with open('data/X_train_tfidf.pickle', 'wb') as output:
    pickle.dump(X_train_tfidf, output)

with open('data/X_test_tfidf.pickle', 'wb') as output:
    pickle.dump(X_test_tfidf, output)