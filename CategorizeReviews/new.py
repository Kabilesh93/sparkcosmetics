import pandas as pd
import nltk
import re
from collections import Counter
from nltk.corpus import stopwords

df = pd.read_csv('./CSV/perfumeReviews.csv', sep='delimiter', engine="python", error_bad_lines=False)
df.columns = ['Text']
df.Text = df.Text.str.replace('\d+', '')
df.Text = df.Text.map(lambda x: x.lstrip('\t').rstrip('aAbBcC'))

WORD = re.compile(r'\w+')

exclude = {'~', ':', '\\', '|', '+', '}', ')', '>', '/', '@', '&', '<', '^', '-', '{', ';', '!', '[', '=', '_', '#', '`', '?', ',', '"', '.', '%', '$', '(', '*', ']'}
tagged_list = []
Category = []
vals = []
tokens = []
reviewsWithFeat = {}
regex = re.compile(r'[\n\r\t]')


def extract_feature(review):
    review = review.lower()
    review = re.sub(r'(http|www)\S+', '', review)
    review = re.sub(r'\d+', '', review)
    review = regex.sub(" ", review)
    review = review.split(" ")
    review = " ".join([word for word in review if word not in stopwords.words('english')])
    review = "".join(ch for ch in review if ch not in exclude)
    flist = []
    words = []
    tokens = nltk.word_tokenize(review)
    tagged = nltk.pos_tag(tokens)

    if len(tokens) > 1:
        for i in range(len(tagged) - 1):
            if i+2 <= len(tagged)-1:
                if (
                        tagged[i][1] == 'JJ' and (tagged[i + 1][1] == 'NN' or tagged[i + 1][1] == 'NNS')
                        or (tagged[i][1] == 'RB' or tagged[i][1] == 'RBR' or tagged[i][1] == 'RBS') and tagged[i + 1][1] == 'JJ' and not (tagged[i + 2][1] == 'NN' or tagged[i + 2][1] == 'NNS')
                        or tagged[i][1] == 'JJ' and tagged[i + 1][1] == 'JJ' and not (tagged[i + 2][1] == 'NN' or tagged[i + 2][1] == 'NNS')
                        or (tagged[i][1] == 'NN' or tagged[i][1] == 'NNS') and tagged[i + 1][1] == 'JJ' and not (tagged[i + 2][1] == 'NN' or tagged[i + 2][1] == 'NNS')
                        or (tagged[i][1] == 'RB' or tagged[i][1] == 'RBR' or tagged[i][1] == 'RBS') and (tagged[i + 1][1] == 'VB' or tagged[i + 1][1] == 'VBD' or tagged[i + 1][1] == 'VBN' or tagged[i + 1][1] == 'VBG')
                ):
                    flist.append(tagged[i][0] + " " + tagged[i + 1][0])
            elif (
                    tagged[i][1] == 'JJ' and (tagged[i + 1][1] == 'NN' or tagged[i + 1][1] == 'NNS')
                    or (tagged[i][1] == 'RB' or tagged[i][1] == 'RBR' or tagged[i][1] == 'RBS') and (tagged[i + 1][1] == 'VB' or tagged[i + 1][1] == 'VBD' or tagged[i + 1][1] == 'VBN' or tagged[i + 1][1] == 'VBG')
            ):
                flist.append(tagged[i][0] + " " + tagged[i + 1][0])

    return flist


i=1
for review_text in df.Text:
    feature = extract_feature(review_text)
    [vals.append(word) for word in feature]
    print(i)
    i=i+1

count = Counter(vals)
min_support = int((len(df["Text"])/100)*2)
categories = []
for category in count.most_common():
    if category[1] >= min_support:
        categories.append(category)
    else:
        break


def categorize(reviewText):
    featureList = extract_feature(reviewText)
    c = "Neutral"
    for categ in categories:
        if categ[0] in featureList:
            c = categ[0]
    return c


StackValues = df.Text.apply(lambda x: categorize(x))
df['category'] = StackValues
df.Text = df.Text.str.replace('[",]', '')
df.to_csv("CSV/CategorisedPerfumeReviews.csv", index=False)
print(len(categories))
