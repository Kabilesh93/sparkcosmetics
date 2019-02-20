from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

def imagePridiction(sample_input, collected_data):

    dirname = os.path.dirname(__file__)

# Load dataset
    data1 = collected_data

# Define the categories
    X = data1[['Ad1 [Product is displayed in smaller size]',
               'Ad1 [Ad with the brand name]',
               'Ad1 [Displaying the subsidiaries of the same brand]',
               'Ad1 [Model wearing the product]',
               'Ad1 [Showing the outcome of using the product]']]  # Features
    y = data1['Ad1 [How much you like this ad]']  # Labels

    # sample_input = [[1, 1, 1, 1, 1]]

#   Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0) # 70% training and 30% test


#   Train with RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    y_pred2 = clf.predict(sample_input)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(metrics.accuracy_score(y_test, y_pred))
    print(y_pred2)

#     Train with SVM
#     SVM = svm.LinearSVC()
#     SVM.fit(X_train, y_train)
#     y_pred = SVM.predict(X_test)
#     y_pred2 = SVM.predict(sample_input)
#
#     print(confusion_matrix(y_test, y_pred))
#     print(classification_report(y_test, y_pred))
#     print(accuracy_score(y_test, y_pred))
#     print(y_pred2)

    return y_pred2

# # sample_input = [[5, 1, 1, 5, 5]]
# out = imagePridiction(sample_input)
# # print(out)