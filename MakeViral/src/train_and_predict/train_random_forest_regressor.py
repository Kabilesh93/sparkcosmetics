import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor

dirname = os.path.dirname(__file__)


def train_retweet_count_prediction(dataset):

    tfidf = TfidfVectorizer(stop_words='english', lowercase=False)

    keyword_response = tfidf.fit_transform(dataset['text'])
    pd.set_option('display.max_colwidth', -1)
    keyword_matrix = pd.DataFrame(keyword_response.todense(), columns=tfidf.get_feature_names())
    keyword_result = keyword_matrix.loc[:, (keyword_matrix.sum() >= 20)]
    keyword_result_for_syns = keyword_matrix.loc[:, (keyword_matrix.sum() >= 10)]

    dataset = dataset.drop(['text', 'text_posTagged'], axis=1)
    result = pd.concat([keyword_result, dataset], axis=1, sort=False)

    result = result.drop(['favourite_count'], axis=1)
    result = result.dropna()

    X = result.loc[:, result.columns != 'retweet_count']
    y = result['retweet_count']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.00005, random_state=0)

    regressor = RandomForestRegressor(n_estimators=30, random_state=0)
    regressor.fit(X_train, y_train)

    # and later you can load it
    joblib.dump(regressor, os.path.join(dirname, '../../knowledge_base/knowledge_retweet_count.pkl'))

    X_test.to_csv(os.path.join(dirname, '../../data/X_test_rt.csv'), sep='\t', encoding='utf-8', index=False)

    keyword_result_for_syns.head(0).to_csv(os.path.join(dirname, '../../data/keywords.csv'), sep='\t', encoding='utf-8', index=False)

    return None


def train_favourite_count_prediction(dataset):

    tfidf = TfidfVectorizer(stop_words='english', lowercase=False)

    keyword_response = tfidf.fit_transform(dataset['text'])
    pd.set_option('display.max_colwidth', -1)
    keyword_matrix = pd.DataFrame(keyword_response.todense(), columns=tfidf.get_feature_names())
    keyword_result = keyword_matrix.loc[:, (keyword_matrix.sum() >= 20)]

    dataset = dataset.drop(['text', 'text_posTagged'], axis=1)
    result = pd.concat([keyword_result, dataset], axis=1, sort=False)

    result = result.drop(['retweet_count'], axis=1)
    result = result.dropna()

    X = result.loc[:, result.columns != 'favourite_count']
    y = result['favourite_count']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.00005, random_state=0)

    regressor = RandomForestRegressor(n_estimators=30, random_state=0)
    regressor.fit(X_train, y_train)

    # and later you can load it
    joblib.dump(regressor, os.path.join(dirname, '../../knowledge_base/knowledge_favourite_count.pkl'))

    X_test.to_csv(os.path.join(dirname, '../../data/X_test_fv.csv'), sep='\t', encoding='utf-8', index=False)

    return None





