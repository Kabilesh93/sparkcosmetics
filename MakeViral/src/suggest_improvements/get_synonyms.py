from nltk.corpus import wordnet
import itertools as it


def getSynonyms(tweet_preprocessed, keywords):

    keywords_list = list(keywords.columns.values)
    #
    columns_in_tweet = list(tweet_preprocessed.columns.values)

    keywords_in_tweet = []
    for x in columns_in_tweet:
        if (x == 'hour'):
            break
        keywords_in_tweet.append(x)

    syns_for_keywords = dict()
    for x in keywords_in_tweet:
        key = x
        syns_for_keywords[key] = []
        syns = wordnet.synsets(x)
        for x in syns:
          for y in x.lemmas():
              if y.name() in keywords_list:
                syns_for_keywords[key].append(y.name())
    #
    for lst in syns_for_keywords.values():
        lst[:] = list(set(lst))
    #
    for key, value in syns_for_keywords.items():
        if (syns_for_keywords.get(key)==[]):
            syns_for_keywords[key] = [key]
    #
    allWords = sorted(syns_for_keywords)
    combinations = it.product(*(syns_for_keywords[Word] for Word in allWords))
    keyword_combinations = list(combinations)
    result = []
    for x in keyword_combinations:
        result.append(' '.join(word for word in list(x)))

    return result