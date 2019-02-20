import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import emoji
from ast import literal_eval
import string
from nltk import *
import os
import numpy as np
from time import gmtime, strftime

dirname = os.path.dirname(__file__)


def preprocess_tweet(text, dataset):

    hasHashtag = 1 if re.search(r"#(\w+)", text) else 0
    date_time = strftime("%Y-%m-%d %H:%M:%S")

    tweet = {'id': ['0000001'], 'created_at': [date_time],
             'text': [text],
             'hasMedia': [1], 'hasHashtag': [hasHashtag], 'followers_count': [345],
             'retweet_count': [dataset['retweet_count'].mean()], 'favourite_count': [dataset['favourite_count'].mean()]}

    tweet_df = pd.DataFrame(data=tweet)

    created_at_Splitted = tweet_df['created_at'].str.split(' ', 1, expand=True).rename(
        columns={0: 'date', 1: 'time'})
    concatinated = pd.concat([tweet_df, created_at_Splitted], axis=1)
    created_at_dropped = concatinated.drop(['created_at'], axis=1)
    reordered = created_at_dropped[
        ["id", "date", "time", "text", "hasMedia", "hasHashtag", "followers_count", "retweet_count",
         "favourite_count"]].copy()

    reordered['date'] = reordered['date'].str.slice(start=5, stop=7, step=None)
    reordered['time'] = reordered['time'].str.slice(start=0, stop=2, step=None)

    reordered[['date','time']] = reordered[['date','time']].apply(pd.to_numeric)

    reordered.rename(columns={'date': 'month', 'time': 'hour'}, inplace=True)

    tweet_df = reordered

    tweet_df['text'] = tweet_df['text'].str.replace('http\S+|www.\S+', '', case=False)
    tweet_df['text'] = tweet_df['text'].str.replace('#\S+', '', case=False)
    tweet_df['text'] = tweet_df['text'].str.replace('@\S+', '', case=False)

    analyzer = SentimentIntensityAnalyzer()
    tweet_df['sentiments'] = tweet_df.apply(lambda row: analyzer.polarity_scores(row['text']), axis=1)

    emojis_list = map(lambda x: ''.join(x.split()), emoji.UNICODE_EMOJI.keys())
    r = re.compile('|'.join(re.escape(p) for p in emojis_list))
    tweet_df['emojis'] = tweet_df.apply(lambda row: ' '.join(r.findall(row['text'])), axis=1)
    tweet_df['emojis'] = [0 if len(x) == 0 else 1 for x in tweet_df['emojis']]

    tweet_df['text'] = tweet_df['text'].str.replace(r'[^\x00-\x7F]+', '', regex=True)

    tweet_df['text'] = tweet_df['text'].apply(emoji.demojize)

    punctuations = string.punctuation
    punctuations = punctuations.replace(',', '')
    tweet_df['text'] = tweet_df['text'].apply(sent_tokenize)
    f = lambda sent: ''.join(ch for w in sent for ch in w
                             if ch not in string.punctuation)

    tweet_df['text'] = tweet_df['text'].apply(lambda row: list(map(f, row)))

    tweet_df['text'] = tweet_df['text'].astype(str).str.lower().transform(literal_eval)

    tweet_df['text'] = tweet_df['text'].apply(lambda row: list(map(word_tokenize, row)))

    tweet_df['text_posTagged'] = tweet_df['text'].apply(lambda row: list(map(pos_tag, row)))

    tweet_df['text_posTagged'] = tweet_df['text_posTagged'].apply(
        lambda row: [item for sublist in row for item in sublist])

    tweet_df.text_posTagged = tweet_df.text_posTagged.apply(lambda x: [(t[1]) for t in x])
    tweet_df.ix[tweet_df.text_posTagged.apply(len) == 0, 'text_posTagged'] = [[np.nan]]

    anger_dict = open(os.path.join(dirname, '../../dicts/emotions_dictionary/anger.txt'), "r")
    anger_words = anger_dict.read().splitlines()

    anticipation_dict = open(os.path.join(dirname, '../../dicts/emotions_dictionary/anticipation.txt'), "r")
    anticipation_words = anticipation_dict.read().splitlines()

    disgust_dict = open(os.path.join(dirname, '../../dicts/emotions_dictionary/disgust.txt'), "r")
    disgust_words = disgust_dict.read().splitlines()

    fear_dict = open(os.path.join(dirname, '../../dicts/emotions_dictionary/fear.txt'), "r")
    fear_words = fear_dict.read().splitlines()

    joy_dict = open(os.path.join(dirname, '../../dicts/emotions_dictionary/joy.txt'), "r")
    joy_words = joy_dict.read().splitlines()

    sadness_dict = open(os.path.join(dirname, '../../dicts/emotions_dictionary/sadness.txt'), "r")
    sadness_words = sadness_dict.read().splitlines()

    surprise_dict = open(os.path.join(dirname, '../../dicts/emotions_dictionary/surprise.txt'), "r")
    surprise_words = surprise_dict.read().splitlines()

    trust_dict = open(os.path.join(dirname, '../../dicts/emotions_dictionary/trust.txt'), "r")
    trust_words = trust_dict.read().splitlines()

    emotion_list = []
    for tweet in tweet_df['text']:
        anger = anticipation = disgust = fear = joy = sadness = surprise = trust = 0
        for sentence in tweet:
            for word in sentence:
                if word in anger_words:
                    anger += 1
                if word in anticipation_words:
                    anticipation += 1
                if word in disgust_words:
                    disgust += 1
                if word in fear_words:
                    fear += 1
                if word in joy_words:
                    joy += 1
                if word in sadness_words:
                    sadness += 1
                if word in surprise_words:
                    surprise += 1
                if word in trust_words:
                    trust += 1
        emotion_values = {
            "anger": anger,
            "anticipation": anticipation,
            "disgust": disgust,
            "fear": fear,
            "joy": joy,
            "sadness": sadness,
            "surprise": surprise,
            "trust": trust
        }
        emotion_list.append(emotion_values)
    emotions = pd.DataFrame({'emotions': emotion_list})
    result = pd.concat([tweet_df, emotions], axis=1, sort=False)
    result = result[
        ['id', 'month', 'hour', 'text', 'hasMedia', 'hasHashtag', 'followers_count', 'sentiments', 'emojis', 'emotions',
         'text_posTagged', 'retweet_count', 'favourite_count']]

    result = pd.concat([result.drop(['sentiments'], axis=1), result['sentiments'].apply(pd.Series)], axis=1)
    result = pd.concat([result.drop(['emotions'], axis=1), result['emotions'].apply(pd.Series)], axis=1)
    result = result.drop(['id', 'month', 'neg', 'neu', 'pos'], axis=1)
    result.rename(columns={'compound': 'sentiment'}, inplace=True)

    tfidf = TfidfVectorizer(stop_words='english', lowercase=False)

    result['text'] = result['text'].apply(
        lambda row: [item for sublist in row for item in sublist])
    result['text'] = result['text'].str.join(' ')
    result['text_posTagged'] = result['text_posTagged'].str.join(' ')

    result = pd.concat([result] * 3, ignore_index=True)
    keyword_response = tfidf.fit_transform(result['text'])
    pd.set_option('display.max_colwidth', -1)
    keyword_matrix = pd.DataFrame(keyword_response.todense(), columns=tfidf.get_feature_names())

    tfidf_result = pd.concat([keyword_matrix], axis=1, sort=False)
    result = result.drop(['text'], axis=1)
    result = pd.concat([tfidf_result, result], axis=1, sort=False)
    result = result.head(1)

    return result