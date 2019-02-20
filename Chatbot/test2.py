from test1 import get_entity1
import nltk
from textblob.classifiers import NaiveBayesClassifier
import csv
import time

a, b, c, d, e, f, g = get_entity1()


def extract_entity(str):
  sentence = str.lower()

  # Tokenize the words
  tokenized=nltk.word_tokenize(sentence)

  # Function to test if something is a noun
  is_noun = lambda pos: pos[:2] == 'NN'
  nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)]
  #print("Nouns identified :",nouns)

  # Function to test if something is a verb
  is_verb = lambda pos: pos[:2] == 'VB'
  verbs = [word for (word, pos) in nltk.pos_tag(tokenized) if is_verb(pos)]

  # Function to test if something is a proverb
  is_prp = lambda pos: pos[:2] == 'PR'
  prps = [word for (word, pos) in nltk.pos_tag(tokenized) if is_prp(pos)]

  # Function to test if something is a Questioning words
  is_qw = lambda pos: pos[:2] ==  'WP' or 'WR'
  wps= [word for (word, pos) in nltk.pos_tag(tokenized) if is_qw(pos)]




  if(nouns==[]):

    t="general"
    return t




  else:

    file = open('Entity.csv', 'r')
    reader = csv.reader(file)
    feature_set = []

    for word, label in reader:
        feature_set.append((word, label))

    cl = NaiveBayesClassifier(feature_set)
    entity=cl.classify(nouns)

    return entity


    #
    #
    # if (entity=="1wrd"):
    #     print("It is hiting the 1wrd")
    #     verbs=nouns
    #
    # file = open(entity+'.csv', 'r')
    # reader = csv.reader(file)
    # feature_set = []
    #
    # for intent,response in reader:
    #     feature_set.append((intent, response))
    #
    # print(feature_set)
    #
    # cl = NaiveBayesClassifier(feature_set)
    # print(verbs,"The response is:")
    # res=cl.classify(verbs)
    #
    # t=res
    #return t


# question = input("Enter the question related to cosmetics :")
# start_time = time.time()
# x = extract_entity(question)
# #print("Entity identified :",x)
#
# if(x == "soap"):
#     start_time = time.time()
#     response = a.get_response(question)
#     print("Response :",response)
#
# print("Runtime : "+" %.3f Seconds " % float(time.time() - start_time))

