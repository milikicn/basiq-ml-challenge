from sklearn.feature_extraction.text import TfidfVectorizer


class BasiqTfidfVectorizer():

    def __init__(self):
        # configure the TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(encoding='utf-8',
                                                ngram_range=(1, 3),  # we want to create unigrams and bigrams
                                                stop_words=None,  # already applied
                                                lowercase=False,  # already applied
                                                max_df=0.95, # remove all terms that have document frequency lower than 5th percentile
                                                min_df=0.05, # remove all terms that have document frequency higher than 95th percentile
                                                max_features=500,
                                                norm='l2',
                                                sublinear_tf=True)

    def fit_transform(self, train_data):
        return self.tfidf_vectorizer.fit_transform(train_data).toarray()

    def transform(self, data):
        return self.tfidf_vectorizer.transform(data).toarray()

    def get_feature_names(self):
        return self.tfidf_vectorizer.get_feature_names()
