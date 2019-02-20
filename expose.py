import random

from flask import Flask, jsonify, request
from flask_cors import CORS
from scipy import stats

from MakeViral.src.get_data.get_data import *
from MakeViral.src.image_relavance.measure_image_relavance import *
from MakeViral.src.preprocess.emotion_extraction import *
from MakeViral.src.preprocess.preprocess import *
from MakeViral.src.preprocess.preprocess_tweet import preprocess_tweet
from MakeViral.src.suggest_improvements.get_synonyms import *
from MakeViral.src.suggest_improvements.improve_sentiment import *
from MakeViral.src.suggest_improvements.predict_count_for_syns import *
from MakeViral.src.train_and_predict.measure_performance import *
from MakeViral.src.train_and_predict.predict_counts import *
from MakeViral.src.train_and_predict.train_random_forest_regressor import *

from nltk.corpus import stopwords
import pandas as pd
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask,request,jsonify
from flask_cors import CORS

from Chatbot.test1 import get_entity1
from Chatbot.test22 import extract_entity1


app = Flask(__name__)
CORS(app)


@app.route("/reviews", methods=["POST", "GET"])
def get_responses():
    df = pd.read_csv('CategorizeReviews/CSV/CategorisedPerfumeReviews.csv', sep=',', engine="python", error_bad_lines=False)
    df["Text"] = df["Text"].str.replace('[,]', ' ')

    categories = list(set(list(df["category"])))
    review_dict = {}

    for category in categories:
        review_dict[category] = []

    review_list = []

    for i in df.values.tolist():
        for cate in categories:
            if cate in i[1]:
                temp = review_dict[cate]
                temp.append(i[0])
                break

    for c in categories:
        new_list = [c] + review_dict[c]
        review_list.append(new_list)
    return jsonify(review_list)


@app.route("/predict", methods=["POST", "GET"])
def prediction():
    str = request.get_json()
    r = str.get("message", "none")
    review = [r]
    print(review)

    tfidfconverter = TfidfVectorizer(stop_words=stopwords.words('english'), max_features=4500)
    X_test = pd.read_csv('CategorizeReviews/CSV/X_test.csv')
    X_new = tfidfconverter.fit_transform(review)
    keyword_matrixs = pd.DataFrame(X_new.todense(), columns=tfidfconverter.get_feature_names())
    missing_cols = set(X_test.columns) - set(keyword_matrixs.columns)

    for c in missing_cols:
        keyword_matrixs[c] = 0

    keyword_matrixs = keyword_matrixs[X_test.columns]

    SVM = joblib.load('CategorizeReviews/CSV/trained_model.pkl')
    y_pred = SVM.predict(keyword_matrixs)
    print(''.join(y_pred))
    return jsonify(''.join(y_pred))




@app.route("/chat", methods=["POST","GET"])
def get_response():
    str=request.get_json()
    response=''
    question = str.get("message","none")


    a, b, c, d, e, f, g = get_entity1()
    entity = extract_entity1(question)
    #print("entity is:",entity)

    if (entity == "soap"):
        response = a.get_response(question).text

    else:
        if (entity == "shampoo"):
            response = b.get_response(question).text

        else:
            if (entity == "powder"):
                response = c.get_response(question).text

            else:
                if (entity == "lipstick"):
                    response = d.get_response(question).text

                else:
                    if (entity == "perfume"):
                        response = e.get_response(question).text
                    else:
                        if (entity == "general"):
                            #print(entity)
                            response = g.get_response(question).text
                        else:
                            response="Sorry My questions are Limited to Cosmetic Domain"
#reinforcement learning
    with open("Chatbot/" + entity + '.yml') as myfile:
        if not ("- - " + question) in myfile.read():
            file = open("Chatbot/updated.yml", "a")
            file.writelines("- - " + question + "\n")
            file.writelines("  - " + response + "\n")
            file.close()
    # print(response.text)

    return jsonify(response)








def load_data():
    fullcorpus = pd.read_csv("MakeViral/data/tweets.csv")
    fullcorpus.columns = ["id", "created_at", "text", "hasMedia", "hasHashtag", "followers_count", "retweet_count",
                          "favourite_count"]

    return fullcorpus


def load_preprocessed():
    preprocessed = pd.read_csv(os.path.join(dirname, 'MakeViral/data/preprocessed_data.csv'), sep='\t')
    return preprocessed


def load_data_for_training():
    dataset = pd.read_csv('MakeViral/data/data_for_training.csv', sep='\t')
    result = dataset.drop(['text', 'text_posTagged'], axis=1)
    result = result[(np.abs(stats.zscore(result)) < 15).all(axis=1)]
    result = pd.concat([result, dataset['text'], dataset['text_posTagged']], axis=1, sort=False)
    return result


def load_xtest_rt():
    result = pd.read_csv('MakeViral/data/X_test_rt.csv', sep='\t')
    return result


def load_xtest_fv():
    result = pd.read_csv('MakeViral/data/X_test_fv.csv', sep='\t')
    return result


def load_keywords():
    result = pd.read_csv('MakeViral/data/keywords.csv', sep='\t')
    return result


def load_positive():
    positive_words = open('MakeViral/dicts/emotions_dictionary/positive.txt', "r")
    positive_words = positive_words.read().splitlines()
    return positive_words


def load_negative():
    negative_words = open('MakeViral/dicts/emotions_dictionary/negative.txt', "r")
    negative_words = negative_words.read().splitlines()
    return negative_words


def data_collected():
    collected_data = pd.read_csv('MakeViral/data/data_collected.csv', header=0)
    return collected_data


@app.route("/getData", methods = ['POST', 'GET'])
def getData():
    # get_all_tweets("@tartecosmetics")

    # @tartecosmetics 6381
    # @ABHcosmetics 12821
    # @BenefitBeauty 19225
    # @NyxCosmetics 25693
    # @UrbanDecay 32141
    # @NARSCosmetics 38557
    # @MAKEUPFOREVERUS 44967
    # @wetnwildbeauty 51371
    # @MACcosmetics 57795

    return 'x'


@app.route("/preprocess", methods = ['POST', 'GET'])
def preprocess():
    fullcorpus = load_data()
    preparedData = prepareData(fullcorpus)
    preprocessedData = preprocessData(preparedData)

    return 'x'


@app.route("/generate_data_for_training", methods = ['POST', 'GET'])
def generate_data_for_training():
    preprocessed = load_preprocessed()
    emotions = list_emotions(preprocessed)

    return 'x'

@app.route("/train_models", methods = ['POST', 'GET'])
def train_models():
    dataset = load_data_for_training()

    train_retweet_count_prediction(dataset)
    train_favourite_count_prediction(dataset)

    return 'x'


@app.route("/predict_initial", methods = ['POST', 'GET'])
def predict_initial():

    dataset = load_data_for_training()

    strg = request.get_json()
    tweet = strg.get("tweet", "none").get("text", "none")

    tweet_preprocessed = preprocess_tweet(tweet, dataset)

    X_test_rt = load_xtest_rt()
    print('Initial number of retweets')
    retweet_count = retweet_count_prediction(X_test_rt, tweet_preprocessed)
    print(retweet_count)

    X_test_fv = load_xtest_fv()
    print('Initial number of favorites')
    favourite_count = favourite_count_prediction(X_test_fv, tweet_preprocessed)
    print(favourite_count)
    print('\n')

    popularity = (retweet_count * 20) + (favourite_count * 1) / dataset.shape[0]

    print('Popularity' , popularity)

    retweets = int(pd.Series(retweet_count)[0])
    favorites = int(pd.Series(favourite_count)[0])
    initial_popularity = int(pd.Series(popularity)[0])
    hasHashtag = int(tweet_preprocessed.loc[0]['hasHashtag'])
    hasMedia = int(tweet_preprocessed.loc[0]['hasMedia'])
    emojis = int(tweet_preprocessed.loc[0]['emojis'])
    pos_tag = tweet_preprocessed.loc[0]['text_posTagged']

    actual_tweet = []
    for x in list(tweet_preprocessed.columns.values):
        if x == 'hour':
            break
        actual_tweet.append(x)

    actual_tweet = ' '.join(actual_tweet)

    return jsonify(initial_retweets=retweets, initial_favorites=favorites, initial_popularity=initial_popularity,
                   hasHashtag=hasHashtag, hasMedia=hasMedia, emojis=emojis, pos_tag=pos_tag)


@app.route("/improve_tweet", methods = ['POST', 'GET'])
def improve_tweet():

    dataset = load_data_for_training()
    tweet = request.get_json()
    text = tweet.get("tweet", "none").get("text", "none")

    initial_reach = tweet.get("initial_reach", "none")
    initial_retweet_count = tweet.get("initial_retweet_count", "none")
    initial_favourite_count = tweet.get("initial_favourite_count", "none")

    tweet_preprocessed = preprocess_tweet(text, dataset)

    X_test_rt = load_xtest_rt()
    X_test_fv = load_xtest_fv()

    keywords = load_keywords()
    synonyms_result = getSynonyms(tweet_preprocessed, keywords)
    if len(synonyms_result) >= 50:
        synonyms_result = random.sample(synonyms_result, 50)

    print('Replacing words with synonyms')
    best_synonyms_list_kw, max_reach_kw, retweet_count_kw, favorite_count_kw = \
        reach_prediction_forSyns(dataset, X_test_rt, X_test_fv, synonyms_result, initial_retweet_count, initial_favourite_count)
    print('modified tweet with highest reach : ', best_synonyms_list_kw)
    print('highest reach : ', max_reach_kw)
    print('highest reach : ', retweet_count_kw)
    print('highest reach : ', favorite_count_kw)
    print('\n')


    #
    print('Improving Sentiment\n')
    positive_words = load_positive()
    negative_words = load_negative()
    #
    sentImprovedWithSyns = get_syns_for_neg_words(positive_words, negative_words, keywords, best_synonyms_list_kw)

    best_synonyms_list_s, max_reach_s, retweet_count_s, favorite_count_s = \
        reach_prediction_forSyns(dataset, X_test_rt, X_test_fv, sentImprovedWithSyns, initial_retweet_count, initial_favourite_count)
    print('modified tweet with highest reach : ', best_synonyms_list_s)
    print('highest reach : ', max_reach_s)
    print('highest reach : ', retweet_count_s)
    print('highest reach : ', favorite_count_s)

    print('\n')
    #
    #
    actual_tweet = []
    for x in list(tweet_preprocessed.columns.values):
        if x == 'hour':
            break
        actual_tweet.append(x)

    actual_tweet = ' '.join(actual_tweet)
    #
    suggested_keywords = ''
    suggested_sent_improvement = ''
    suggested_text = ''

    if (initial_reach >= max_reach_s)  and (initial_reach >= max_reach_kw):
        suggested_text = actual_tweet
        enhanced_retweet_count = int(pd.Series(initial_retweet_count)[0])
        enhanced_favorite_count = int(pd.Series(initial_favourite_count)[0])
        popularity = initial_reach
    elif (max_reach_kw >= initial_reach)  and (max_reach_kw >= max_reach_s):
        suggested_text = best_synonyms_list_kw
        enhanced_retweet_count = int(pd.Series(retweet_count_kw)[0])
        enhanced_favorite_count = int(pd.Series(favorite_count_s)[0])
        popularity = max_reach_kw
    else:
        suggested_text = best_synonyms_list_s
        enhanced_retweet_count = int(pd.Series(retweet_count_s)[0])
        enhanced_favorite_count = int(pd.Series(favorite_count_s)[0])
        popularity = max_reach_s

    print('Initial Text : ', actual_tweet)
    print('Suggested Text : ',suggested_text)

    max_reach_kw = pd.Series(max_reach_kw)[0]
    max_reach_s = pd.Series(max_reach_s)[0]
    enhanced_retweet_count = int(pd.Series(enhanced_retweet_count)[0])
    enhanced_favorite_count = int(pd.Series(enhanced_favorite_count)[0])

    if max_reach_kw > initial_reach:
        suggested_keywords = best_synonyms_list_kw
    else:
        suggested_keywords = actual_tweet

    if max_reach_s > initial_reach:
        suggested_sent_improvement = best_synonyms_list_s
    else:
        suggested_sent_improvement = actual_tweet

    return jsonify(enhanced_popularity=int(popularity), enhanced_retweet_count=enhanced_retweet_count,
                   enhanced_favorite_count=enhanced_favorite_count, actual_tweet=actual_tweet, suggested_tweet=suggested_text,
                   suggested_keywords=suggested_keywords, suggested_sent_improvement=suggested_sent_improvement)


@app.route("/measure_performance", methods=['POST', 'GET'])
def measure_performance():

    dataset = load_data_for_training()

    predict_retweets(dataset)
    predict_favorites(dataset)

    return jsonify('dsbhjgds')


@app.route("/image_relavance", methods=['POST', 'GET'])
def image():

    collected_data = data_collected()
    values = request.get_json()
    image_values = values.get("values", "none")
    image_values = list(map(int, image_values))

    result = imagePridiction([image_values], collected_data)
    result_pcn = int(pd.Series(result)[0]) * 20
    return jsonify(image_relavance=result_pcn)


if __name__ == "__main__":
   app.run(host='127.0.0.1', port=5002, debug=True)
