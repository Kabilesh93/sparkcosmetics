import tweepy
import csv
import os

dirname = os.path.dirname(__file__)

# Twitter API credentials
consumer_key = "cqCJE5WMcUQ0mJ0D141cDnsPs"
consumer_secret = "hnuIWr3aoStiJCIgxjhLoO3cVSb8qPsmwfsioVeLNVb4N3M8sR"
access_key = "3018072521-1uwUUzdrUOkW5wn0KhBp2u9JeneSsyLcQ2KeMN4"
access_secret = "qRr8j8rDsutbEftqfnEM5CQOyfJUnW902XQU8iNEma8lw"


def get_all_tweets(screen_name):
    # Twitter only allows access to a users most recent 3240 tweets with this method

    # authorize twitter, initialize tweepy
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)

    # initialize a list to hold all the tweepy Tweets
    alltweets = []

    # make initial request for most recent tweets (200 is the maximum allowed count)
    new_tweets = api.user_timeline(screen_name=screen_name, count=200, include_entities=True, tweet_mode='extended')

    # save most recent tweets
    alltweets.extend(new_tweets)

    # save the id of the oldest tweet less one
    oldest = alltweets[-1].id - 1

    # keep grabbing tweets until there are no tweets left to grab
    while len(new_tweets) > 0:
        print
        "getting tweets before %s" % (oldest)

        # all subsiquent requests use the max_id param to prevent duplicates
        new_tweets = api.user_timeline(screen_name=screen_name, count=200, max_id=oldest, include_entities=True,
                                       tweet_mode='extended')

        # save most recent tweets
        alltweets.extend(new_tweets)

        # update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1

        print
        "...%s tweets downloaded so far" % (len(alltweets))

    user = api.get_user(screen_name)
    followers_count = user.followers_count

    # # transform the tweepy tweets into a 2D array that will populate the csv
    outtweets = [[tweet.id_str, tweet.created_at,  tweet.text, 1 if 'media' in tweet.entities else 0,
                  1 if tweet.entities.get('hashtags') else 0, followers_count, tweet.retweet_count, tweet.favorite_count]
                 for tweet in alltweets]

    # write the csv
    with open(os.path.join(dirname, '../../data/tweets.csv'), mode='a', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(outtweets)

    pass

    return None
