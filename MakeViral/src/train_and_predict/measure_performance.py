import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt


def predict_retweets(dataset):

    print('Evaluation of retweet count prediction')
    dataset = dataset.head(20000)

    tfidf = TfidfVectorizer(stop_words='english', lowercase=False)

    keyword_response = tfidf.fit_transform(dataset['text'])
    pd.set_option('display.max_colwidth', -1)
    keyword_matrix = pd.DataFrame(keyword_response.todense(), columns=tfidf.get_feature_names())
    keyword_matrix = keyword_matrix.loc[:, (keyword_matrix.sum() >= 20)]

    pos_tag_response = tfidf.fit_transform(dataset['text_posTagged'])
    pd.set_option('display.max_colwidth', -1)
    pos_tag_matrix = pd.DataFrame(pos_tag_response.todense(), columns=tfidf.get_feature_names())

    # tfidf_result = pd.concat([keyword_matrix, pos_tag_matrix], axis=1, sort=False)
    tfidf_result = pd.concat([keyword_matrix], axis=1, sort=False)
    dataset = dataset.drop(['text', 'text_posTagged'], axis=1)
    result = pd.concat([tfidf_result, dataset], axis=1, sort=False)

    result = result.drop(['favourite_count'], axis=1)
    result = result.dropna()

    X = result.loc[:, result.columns != 'retweet_count']
    y = result['retweet_count']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    regressors = [
        LinearRegression(),
        RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100),
        MLPRegressor(hidden_layer_sizes=(15,), max_iter=100000, learning_rate='constant',learning_rate_init=0.001, early_stopping=False),
        SVR(gamma='scale', C=1.0, epsilon=0.5),
        BayesianRidge(alpha_1=1e-04, alpha_2=1e-04, compute_score=False, lambda_1=1e-04, lambda_2=1e-04, n_iter=500),
    ]
    # Logging for Visual Comparison
    log_cols = ["regressor", "Mean Absolute Error", "Root Mean Square Error"]
    log = pd.DataFrame(columns=log_cols)
    for clf in regressors:
        clf.fit(X_train, y_train)
        name = clf.__class__.__name__

        print("=" * 30)
        print(name)

        print('****Results****')
        y_pred = clf.predict(X_test)

        MAE = metrics.mean_absolute_error(y_test, y_pred)
        print("Mean Absolute Error: ", MAE)

        RMSR = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        print('Root Mean Squared Error: ', RMSR)

        log_entry = pd.DataFrame([[name, MAE, RMSR]], columns=log_cols)
        log = log.append(log_entry)

    print("=" * 30)

    print("\n\n")

    return None


def predict_favorites(dataset):

    print('Evaluation of favorite count prediction')
    dataset = dataset.head(200)

    tfidf = TfidfVectorizer(stop_words='english', lowercase=False)

    keyword_response = tfidf.fit_transform(dataset['text'])
    pd.set_option('display.max_colwidth', -1)
    keyword_matrix = pd.DataFrame(keyword_response.todense(), columns=tfidf.get_feature_names())
    keyword_matrix = keyword_matrix.loc[:, (keyword_matrix.sum() >= 20)]

    pos_tag_response = tfidf.fit_transform(dataset['text_posTagged'])
    pd.set_option('display.max_colwidth', -1)
    pos_tag_matrix = pd.DataFrame(pos_tag_response.todense(), columns=tfidf.get_feature_names())

    # tfidf_result = pd.concat([keyword_matrix, pos_tag_matrix], axis=1, sort=False)
    tfidf_result = pd.concat([keyword_matrix], axis=1, sort=False)
    dataset = dataset.drop(['text', 'text_posTagged'], axis=1)
    result = pd.concat([tfidf_result, dataset], axis=1, sort=False)

    result = result.drop(['retweet_count'], axis=1)
    result = result.dropna()

    X = result.loc[:, result.columns != 'favourite_count']
    y = result['favourite_count']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    regressors = [
        LinearRegression(),
        RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100),
        MLPRegressor(hidden_layer_sizes=(15,), max_iter=100000, learning_rate='constant',learning_rate_init=0.001, early_stopping=False),
        SVR(gamma='scale', C=1.0, epsilon=0.5),
        BayesianRidge(alpha_1=1e-04, alpha_2=1e-04, compute_score=False, lambda_1=1e-04, lambda_2=1e-04, n_iter=500),
    ]
    # Logging for Visual Comparison
    log_cols = ["regressor", "Mean Absolute Error", "Root Mean Square Error"]
    log = pd.DataFrame(columns=log_cols)
    for clf in regressors:
        clf.fit(X_train, y_train)
        name = clf.__class__.__name__

        print("=" * 30)
        print(name)

        print('****Results****')
        y_pred = clf.predict(X_test)

        MAE = metrics.mean_absolute_error(y_test, y_pred)
        print("Mean Absolute Error: ", MAE)

        RMSR = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        print('Root Mean Squared Error: ', RMSR)

        log_entry = pd.DataFrame([[name, MAE, RMSR]], columns=log_cols)
        log = log.append(log_entry)

    print("=" * 30)

    return None
