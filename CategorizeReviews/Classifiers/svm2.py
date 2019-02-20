from sklearn import svm
from nltk.corpus import stopwords
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import SVC

cls = OneVsRestClassifier(estimator=SVC(gamma='auto'))

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

df = pd.read_csv('C:/Users/Suganthan/PycharmProjects/Decoders/final_year/CSV/MultipleCategoryPerfume.csv')
stop = set(stopwords.words('english'))

transformer = TfidfVectorizer(lowercase=True, stop_words=stop, max_features=45)
X = transformer.fit_transform(df.Review)

X_train, X_test, y_train, y_test = train_test_split(X, df.iloc[:, 2:64],
                                                    test_size=0.20, random_state=42)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

model.fit(X_train, y_train)

cls.fit(X_train, y_train)

review = ["After many different skin moisturizers for my sensitive skin I stopped at this one to balance my skin type. No perfume and not greasy. My favourite skin moisturizer"]
test_df = pd.DataFrame(review, columns=["Review"])
X_new = transformer.fit_transform(test_df.Review)
# print(X_new)

# print(X_test.shape[1])
# print(X_test_n)
# SVM = svm.SVC()
# SVM.fit(X_train, y_train)
y_pred = cls.predict(X_test)

print(y_pred)
print(confusion_matrix(y_test.values.argmax(axis=1), y_pred.argmax(axis=1)))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
