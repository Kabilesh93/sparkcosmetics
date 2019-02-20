from ast import literal_eval
import pandas as pd
import os

dirname = os.path.dirname(__file__)


def list_emotions(preprocessed):

    preprocessed['text'] = preprocessed.text.apply(lambda x: literal_eval(x))

    anger_dict = open('../../dicts/emotions_dictionary/anger.txt', "r")
    anger_words = anger_dict.read().splitlines()

    anticipation_dict = open('../../dicts/emotions_dictionary/anticipation.txt', "r")
    anticipation_words = anticipation_dict.read().splitlines()

    disgust_dict = open('../../dicts/emotions_dictionary/disgust.txt', "r")
    disgust_words = disgust_dict.read().splitlines()

    fear_dict = open('../../dicts/emotions_dictionary/fear.txt', "r")
    fear_words = fear_dict.read().splitlines()

    joy_dict = open('../../dicts/emotions_dictionary/joy.txt', "r")
    joy_words = joy_dict.read().splitlines()

    sadness_dict = open('../../dicts/emotions_dictionary/sadness.txt', "r")
    sadness_words = sadness_dict.read().splitlines()

    surprise_dict = open('../../dicts/emotions_dictionary/surprise.txt', "r")
    surprise_words = surprise_dict.read().splitlines()

    trust_dict = open('../../dicts/emotions_dictionary/trust.txt', "r")
    trust_words = trust_dict.read().splitlines()
    i = 0
    emotion_list = []
    for tweet in preprocessed['text']:
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
        emotions_values = {
            "anger": anger,
            "anticipation": anticipation,
            "disgust": disgust,
            "fear": fear,
            "joy": joy,
            "sadness": sadness,
            "surprise": surprise,
            "trust": trust
        }
        emotion_list.append(emotions_values)
        i = i+1
        print(i)
    emotions = pd.DataFrame({'emotions': emotion_list})
    result = pd.concat([preprocessed, emotions], axis=1, sort=False)

    result['text'] = result['text'].apply(
        lambda row: [item for sublist in row for item in sublist])
    result['text'] = result['text'].str.join(' ')

    m = result['text_posTagged'].notna()
    result.loc[m, 'text_posTagged'] = (
        result.loc[m, 'text_posTagged'].apply(literal_eval))

    result['text_posTagged'] = result['text_posTagged'].str.join(' ')

    result = result.dropna()

    result['sentiments'] = result['sentiments'].apply(literal_eval)
    result = pd.concat([result.drop(['sentiments'], axis=1), result['sentiments'].apply(pd.Series)], axis=1)
    result = pd.concat([result.drop(['emotions'], axis=1), result['emotions'].apply(pd.Series)], axis=1)
    result = result.drop(['id', 'month', 'neg', 'neu', 'pos'], axis=1)
    result.rename(columns={'compound': 'sentiment'}, inplace=True)

    result = result[
        ['hour', 'text', 'text_posTagged', 'hasMedia', 'hasHashtag', 'followers_count', 'sentiment', 'emojis', 'anger',
         'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust', 'retweet_count', 'favourite_count']]

    result.to_csv(os.path.join(dirname, '../../data/data_for_training.csv'), sep='\t', encoding='utf-8', index=False)

    return None


