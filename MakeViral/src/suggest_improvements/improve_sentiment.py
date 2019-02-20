from nltk.corpus import wordnet
import itertools as it


def get_syns_for_neg_words(positive_words, negative_words, keywords, best_synonyms_list):

    keywords_list = list(keywords.columns.values)

    keywords_in_tweet = best_synonyms_list.split()

    syns_for_keywords = dict()
    for x in keywords_in_tweet:
        key = x
        syns_for_keywords[key] = []
        if x in negative_words:
            syns = wordnet.synsets(x)
            for x in syns:
                for y in x.lemmas():
                    if y.name() in (keywords_list and positive_words):
                        syns_for_keywords[key].append(y.name())
        else: syns_for_keywords[key].append(x)

    for lst in syns_for_keywords.values():
        lst[:] = list(set(lst))

    for key, value in syns_for_keywords.items():
        if syns_for_keywords.get(key) == []:
            syns_for_keywords[key] = [key]
    #
    allWords = sorted(syns_for_keywords)
    combinations = it.product(*(syns_for_keywords[Word] for Word in allWords))
    keyword_combinations = list(combinations)
    result = []
    for x in keyword_combinations:
        result.append(' '.join(word for word in list(x)))

    return result