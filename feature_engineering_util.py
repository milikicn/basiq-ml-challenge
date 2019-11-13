import numpy as np
import string
import nltk
from nltk.corpus import stopwords
import pandas as pd


def lowercase(feature: pd.Series):
    """Lowercase the text."""
    return feature.str.lower()


def replace_date_time_with_tokens(feature: pd.Series):
    """Search for date and time occurrences in the text and replace them with the DATETOKEN and TIMETOKEN tokens."""

    # Create a regex expressions for all date formats found in the dataset. Date can be prefixed with the string "Date"
    # or "Date:".
    expanded_date_regex = "(date(:)?(\\s)?)?\\d{1,2}\\s+(jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|jun(e)?|jul(" \
                          "y)?|aug(ust)?|sep(tember)?|oct(ober)?|nov(ember)?|dec(ember)?)\\s+\\d{4}"
    short_date_regex = "(date(:)?(\\s)?)?[\\d+]{1,2}[- /.][\\d+]{1,2}[- /.][\\d+]{4}"
    # Create a regex expression for the time format found in the dataset
    time_regex = "(time(:)?\\s)?([0-9]{2}:[0-9]{2})(am|pm)?"

    # Replace both date format occurrences (along with prefixes if they are present) with the DATETOKEN
    feature1 = feature.str.replace(expanded_date_regex, "DATETOKEN")
    feature1 = feature1.str.replace(short_date_regex, "DATETOKEN")
    # Replace time occurrences (along with prefix if it is present) with the TIMETOKEN
    feature1 = feature1.str.replace(time_regex, "TIMETOKEN")

    return feature1


def replace_abbreviations(feature: pd.Series):
    """Replace finance specific abbreviations with an expanded word."""

    # This should be expanded with more terms.
    abbreviations_dict = {
        "w/d": "withdrawal",
        "trf": "transfer",
        "tfr": "transfer",
        "tfer": "transfer",
        "pymt": "payment"
    }
    return feature.replace(abbreviations_dict, regex=True)


def remove_punctuations(feature: pd.Series):
    """Remove punctuation symbols."""
    return feature.apply(lambda x: x.translate(str.maketrans(string.punctuation, ' ' * 32)))


def replace_receipts_with_token(feature: pd.Series):
    """Replace receipt number with the RECEIPTTOKEN."""

    # Create a regex expression that searches for e.g. "Receipt 167448"
    receipt_number_regex = "(receipt)\\s+([\\w|*]{5,30})\\b"

    # Replace receipt occurrences with the RECEIPTTOKEN
    return feature.str.replace(receipt_number_regex, "RECEIPTTOKEN")


def replace_long_numbers_with_tokens(feature: pd.Series):
    """Replace long numbers with a NUMBERTOKEN (that could be an indicator of an account number or some id).
    This is to be checked whether it makes sense."""

    # Create a regex expression that searches for numbers with more that 5 digits. Also, support masked digits
    # (i.e. 498691xxxxxx8454) or having any other letter (e.g. 709926s77.2)
    long_number_regex = "[0-9]{5,15}(\\w+)?([0-9]{1,10})"

    # Replace long number occurrences with the NUMBERTOKEN
    return feature.str.replace(long_number_regex, "NUMBERTOKEN")


def lematize(feature: pd.Series):
    """Perform lematization over the text based on the Wordnet corpora."""

    # Downloading wordnet from NLTK
    nltk.download('wordnet')

    # Saving the whitespace tokenizer and WordNet lemmatizer
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    lemmatizer = nltk.stem.WordNetLemmatizer()

    return feature.apply(lambda x: " ".join([lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(x)]))


def remove_stopwords(feature: pd.Series):
    """Remove stopwords (from the NTLK package)."""

    # Download the stop words list
    nltk.download('stopwords')

    # Load the stop words in english
    stop_words = list(stopwords.words('english'))

    # remove stopwords
    return feature.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))


def remove_numbers(feature: pd.Series):
    """Remove stand-alone numbers"""
    return feature.str.replace("\\d+ ", " ")


def remove_short_words(feature: pd.Series):
    """Remove words with one or two letters (assumes they don't bring any value to the classifier)."""
    return feature.str.replace("\\b[\\w]{1,2}\\b", " ")


def trim_whitespace(feature: pd.Series):
    """Trim whitespaces."""
    return feature.str.replace("\\s+", " ")


def preprocess_transaction_description(description: pd.Series):
    """Run the preprocessing steps over the Series object containing transaction descriptions."""

    return description \
        .pipe(lowercase) \
        .pipe(replace_date_time_with_tokens) \
        .pipe(replace_abbreviations) \
        .pipe(remove_punctuations) \
        .pipe(replace_receipts_with_token) \
        .pipe(replace_long_numbers_with_tokens) \
        .pipe(lematize) \
        .pipe(remove_stopwords) \
        .pipe(remove_numbers) \
        .pipe(remove_short_words) \
        .pipe(trim_whitespace)


def create_transaction_type_feature(transaction_amount: pd.Series):
    """Based on the transaction_amount value, create a new variable transaction_type."""
    return np.where(transaction_amount > 0, 'debit', 'credit')
