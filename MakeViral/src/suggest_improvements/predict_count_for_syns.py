import os
from sklearn.externals import joblib
from MakeViral.src.preprocess.preprocess_tweet import *

dirname = os.path.dirname(__file__)


def reach_prediction_forSyns(dataset, X_test_rt, X_test_fv, synonyms_result, initial_retweet_count, initial_favourite_count):

    retweetCount = initial_retweet_count
    favoriteCount = initial_favourite_count
    max_popularity = (retweetCount * 20) + (favoriteCount * 1) / dataset.shape[0]
    best_synonyms_list = synonyms_result[0]
    best = []

    for synonyms_list in synonyms_result:

        synonyms_list_preprocessed = preprocess_tweet(synonyms_list, dataset)

        y_pred_rt = retweetCount_prediction_forSyns(X_test_rt, synonyms_list_preprocessed)

        y_pred_fv = favoriteCount_prediction_forSyns(X_test_fv, synonyms_list_preprocessed)
        #
        popularity = (y_pred_rt * 20) + (y_pred_fv * 1) / dataset.shape[0]

        best.append([synonyms_list, popularity[0]])
        d = pd.DataFrame(best, columns=['tweet', 'reach'])
        d= d.sort_values(by=['reach'])
        print(d)

        if popularity > max_popularity:
            max_popularity = popularity
            best_synonyms_list = synonyms_list
            retweetCount = y_pred_rt
            favoriteCount = y_pred_fv

    return best_synonyms_list, max_popularity, retweetCount, favoriteCount


def retweetCount_prediction_forSyns(X_test_rt, synonyms_list_preprocessed):

    # # and later you can load it
    regressor = joblib.load(os.path.join(dirname, '../../knowledge_base/knowledge_retweet_count.pkl'))
    #
    missing_cols = set(X_test_rt.columns) - set(synonyms_list_preprocessed.columns)
    # Add a missing column in test set with default value equal to 0
    for c in missing_cols:
        synonyms_list_preprocessed[c] = 0
    # Ensure the order of column in the test set is in the same order than in train set
    synonyms_list_preprocessed = synonyms_list_preprocessed[X_test_rt.columns]

    y_pred_rt = regressor.predict(synonyms_list_preprocessed)

    return y_pred_rt


def favoriteCount_prediction_forSyns(X_test_fv, synonyms_list_preprocessed):

    regressor = joblib.load(os.path.join(dirname, '../../knowledge_base/knowledge_favourite_count.pkl'))
    #
    missing_cols = set(X_test_fv.columns) - set(synonyms_list_preprocessed.columns)
    # Add a missing column in test set with default value equal to 0
    for c in missing_cols:
        synonyms_list_preprocessed[c] = 0
    # Ensure the order of column in the test set is in the same order than in train set
    synonyms_list_preprocessed = synonyms_list_preprocessed[X_test_fv.columns]

    y_pred_fv = regressor.predict(synonyms_list_preprocessed)

    return y_pred_fv