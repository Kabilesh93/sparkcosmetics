import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import csv
import emoji
from ast import literal_eval
import string
from nltk import *
import os
import numpy as np

dirname = os.path.dirname(__file__)


def prepareData(fullcorpus):

    fullcorpus = fullcorpus.head(20000)

    print("Removing Duplicates\n")
    duplicates_removed = fullcorpus.drop_duplicates(subset='id', keep='first', inplace=False)

    print("Spliting created_at\n")
    created_at_Splitted = duplicates_removed['created_at'].str.split(' ', 1, expand=True).rename(
        columns={0: 'date', 1: 'time'})
    concatinated = pd.concat([duplicates_removed, created_at_Splitted], axis=1)
    created_at_dropped = concatinated.drop(['created_at'], axis=1)
    reordered = created_at_dropped[
        ["id", "date", "time", "text", "hasMedia", "hasHashtag", "followers_count", "retweet_count", "favourite_count"]].copy()

    reordered['date'] = reordered['date'].str.slice(start=5, stop=7, step=None)
    reordered['time'] = reordered['time'].str.slice(start=0, stop=2, step=None)

    reordered[['date','time']] = reordered[['date','time']].apply(pd.to_numeric)

    reordered.rename(columns={'date': 'month', 'time': 'hour'}, inplace=True)

    return reordered


def preprocessData(cleanedData):
    print("Decoding..................\n")
    cleanedData['text'] = cleanedData['text'].apply(literal_eval).str.decode("utf-8")

    print("Removing URLs hashtags mentions................\n")
    cleanedData['text'] = cleanedData['text'].str.replace('http\S+|www.\S+', '', case=False)
    cleanedData['text'] = cleanedData['text'].str.replace('#\S+', '', case=False)
    cleanedData['text'] = cleanedData['text'].str.replace('@\S+', '', case=False)

    print("Applying sentiment analysis\n")
    analyzer = SentimentIntensityAnalyzer()
    cleanedData['sentiments'] = cleanedData.apply(lambda row: analyzer.polarity_scores(row['text']), axis=1)

    emojis_list = map(lambda x: ''.join(x.split()), emoji.UNICODE_EMOJI.keys())
    r = re.compile('|'.join(re.escape(p) for p in emojis_list))
    cleanedData['emojis'] = cleanedData.apply(lambda row: ' '.join(r.findall(row['text'])), axis=1)
    cleanedData['emojis'] = [0 if len(x) == 0 else 1 for x in cleanedData['emojis']]

    print("Removing Emojis................\n")

    cleanedData['text'] = cleanedData['text'].str.replace(r'[^\x00-\x7F]+', '', regex=True)

    cleanedData['text'] = cleanedData['text'].apply(emoji.demojize)

    print("Sentence tokenizing and removing punctuations................\n")
    punctuations = string.punctuation
    punctuations = punctuations.replace(',','')
    cleanedData['text'] = cleanedData['text'].apply(sent_tokenize)
    f = lambda sent: ''.join(ch for w in sent for ch in w
                                                  if ch not in string.punctuation)

    cleanedData['text'] = cleanedData['text'].apply(lambda row: list(map(f, row)))

    print("Converting to lowercase................\n")
    cleanedData['text'] = cleanedData['text'].astype(str).str.lower().transform(literal_eval)

    print("Word tokenizing................\n")
    cleanedData['text'] = cleanedData['text'].apply(lambda row:list(map(word_tokenize, row)))

    print("POS tagging................\n")

    cleanedData['text_posTagged'] = cleanedData['text'].apply(lambda row: list(map(pos_tag, row)))

    cleanedData['text_posTagged'] = cleanedData['text_posTagged'].apply(
        lambda row: [item for sublist in row for item in sublist])

    cleanedData.text_posTagged = cleanedData.text_posTagged.apply(lambda x: [(t[1]) for t in x])
    cleanedData.ix[cleanedData.text_posTagged.apply(len) == 0, 'text_posTagged'] = [[np.nan]]

    cleanedData.to_csv(os.path.join(dirname, '../../data/preprocessed_data.csv'), sep='\t', encoding='utf-8', index=False)

    print('Done Preprocessing')

    return None

