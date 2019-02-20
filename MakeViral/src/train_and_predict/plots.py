import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

def plot_vs_fv(dataset):


    # plt.scatter(dataset['anger'], dataset['favourite_count'])
    # plt.ylabel('favourite_count')
    # plt.xlabel('anger')
    # plt.show()
    # dataset = dataset[dataset['favourite_count'] < 5000]
    # dataset = dataset[['pos_sequence_count', 'favourite_count']]
    # dataset = (dataset - dataset.mean()) / (dataset.max() - dataset.min())



    # dataset['sad'] = dataset['anger'] + dataset['disgust'] + dataset['fear'] + dataset['sadness']
    # dataset['happy'] = dataset['joy'] + dataset['surprise'] + dataset['trust']
    #
    # dataset = dataset[(dataset[['sad', 'happy']] != 0).all(axis=1)]

    dataset['popularity'] = ((dataset['retweet_count'] * dataset['retweet_count'].mean()) + (dataset['favourite_count'] * dataset['favourite_count'].mean()))/dataset.shape[0]

    # # result = dataset.groupby('hasMedia', 'hasHashtag').mean()
    #
    # # print(result)
    #
    # plt.bar({result.index, dataset['hasHashtag']}, result['popularity'], color="dimgray")
    # plt.xlabel('Presence of media elements')
    # plt.ylabel('Average Popularity')
    # plt.title('Presence of media elements vs Average Popularity', y=1.1)
    # plt.grid()
    # plt.show()
    #
    # # favourite_count
    # # retweet_count

    plotMedia = dataset.groupby('hasMedia').mean()
    plotHashtag = dataset.groupby('hasHashtag').mean()

    # fig, ax = plt.subplots()
    # ax.bar(plotMedia.index, plotMedia['popularity'], label="Presence of media elements", color="dimgray")
    # ax.bar(plotHashtag.index, plotHashtag['popularity'], label="Presence of media hashtags", color="gray")

    ax = plt.subplot(111)
    barMedia = ax.bar(plotMedia.index - 0.1, plotMedia['popularity'], width=0.2, color='dimgray', align='center')
    barHashtag = ax.bar(plotHashtag.index + 0.1, plotHashtag['popularity'], width=0.2, color='k', align='center')
    ax.legend((barMedia, barHashtag), ('Presence of images/ videos', 'Presence of Hashtags'))

    plt.xlabel('Presence of images/videos and hashtags')
    plt.ylabel('Average Popularity')
    plt.title('Presence of images/videos and hashtags vs Average Popularity', y=1.1)
    plt.grid()
    plt.show()



    return None