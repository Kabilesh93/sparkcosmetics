import re

def findBest(fullcorpus):

    fullcorpus['tags'] = fullcorpus.text.str.findall(r'#.*?(?=\s|$)')
    fullcorpus['reach'] = ((fullcorpus['retweet_count'] * fullcorpus['retweet_count'].mean())+
                           (fullcorpus['favourite_count']*fullcorpus['favourite_count'].mean()))/fullcorpus.shape[0]

    fullcorpus = fullcorpus.sort_values(by=['reach'])
    print( fullcorpus[['tags','reach']])
    # print( fullcorpus['reach'].mean())
    return None