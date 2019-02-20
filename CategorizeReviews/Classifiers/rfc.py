from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

df = pd.read_csv('C:/Users/Suganthan/PycharmProjects/Decoders/final_year/CSV/CategorisedPerfumeReviews.csv"')
col = df.keys().tolist()
df.columns = col
stop = set(stopwords.words('english'))

tfidfconverter = TfidfVectorizer(max_features=4500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
X = tfidfconverter.fit_transform(df["Text"]).toarray()
y = df["category"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
