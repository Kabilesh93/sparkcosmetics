import os
from sklearn.externals import joblib
from sklearn import metrics
import numpy as np

dirname = os.path.dirname(__file__)


def retweet_count_prediction(X_test, tweet_preprocessed):

    # and later you can load it
    regressor = joblib.load(os.path.join(dirname, '../../knowledge_base/knowledge_retweet_count.pkl'))

    missing_cols = set(X_test.columns) - set(tweet_preprocessed.columns)
    # Add a missing column in test set with default value equal to 0
    for c in missing_cols:
        tweet_preprocessed[c] = 0
    # Ensure the order of column in the test set is in the same order than in train set
    tweet_preprocessed = tweet_preprocessed[X_test.columns]

    y_pred = regressor.predict(tweet_preprocessed)

    return y_pred


def favourite_count_prediction(X_test, tweet_preprocessed):

    # and later you can load it
    regressor = joblib.load(os.path.join(dirname, '../../knowledge_base/knowledge_favourite_count.pkl'))

    missing_cols = set(X_test.columns) - set(tweet_preprocessed.columns)
    # Add a missing column in test set with default value equal to 0
    for c in missing_cols:
        tweet_preprocessed[c] = 0
    # Ensure the order of column in the test set is in the same order than in train set
    tweet_preprocessed = tweet_preprocessed[X_test.columns]

    y_pred = regressor.predict(tweet_preprocessed)

    return y_pred