"""
The views.py file performs computations and serves views to the web server
The code for the classifier model is in here.
"""

from django.http import HttpResponse
from django.shortcuts import render
import random
import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings("ignore")
import pickle

# renders the home page
def home(request):
    return render(request, 'home.html')

# receives training file input and calls a file handler function
# renders the model training results page once the model is trained
def result(request):
    if request.method == 'POST':
        myFile = request.FILES['fileName']
        fileName = str(request.FILES['fileName'])
        accuracy, precision, recall, cm, clf, label_set, count = handle_uploaded_file(myFile, fileName)
        return render(request, 'result.html', {'fileName':fileName, 'clf':clf, 'accuracy':accuracy, 'precision':precision, 'recall':recall, 'cm':cm, 'label_set':label_set, 'count':count})
    else:
        return HttpResponse("Missing form data.  Don't access this URL directly.  Don't use the back button, only hyperlinks and page buttons.")

# the classification mode training function
def myModel(dataset):
    # compute the set of labels and how many of each there are
    label_set, count = np.unique(dataset.label, return_counts=True)

	# Cleaning the texts by removing non-alphanumeric characters
    # convert to lower case
    # Lemmatizing words
    # removing stopwords
    corpus = []
    for i in range(len(dataset)):
        text = re.sub('[^a-zA-Z0-9]', ' ', dataset['text'][i])
        corpus.append(text)
    del i, text

    # Creating the TfIdf model (convert clean text into numerical vectors)
    from sklearn.feature_extraction.text import TfidfVectorizer
    tv = TfidfVectorizer(stop_words='english', ngram_range=(1, 1), use_idf=False, norm=None)
    X = tv.fit_transform(corpus).toarray()
    y = dataset.iloc[:, 0].values

    # Splitting the dataset into the training set and test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

    # creating a simple multinomial naive Bayes classifier
    # and fitting the model
    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB(alpha=0.5)
    clf.fit(X_train, y_train)

    # Predicting the test set results
    y_pred = clf.predict(X_test)

    # Measuring model performance
    # Making the Confusion Matrix and model performance measures
    # accuracy - true positives + true negatives/ total
    # precision - true positives / (true positives+false positives)
    # recall - true positives / (true positives + false negatives)
    from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)

    pickle.dump(clf, open('model.sav', 'wb'))
    pickle.dump(tv, open('tv.sav', 'wb'))

    return accuracy, precision, recall, cm, clf, label_set, count

# function to handle the uploaded file and read in the data set
def handle_uploaded_file(myFile, fileName):
    dataset = pd.read_table(myFile, sep='\t', header=None, names=['label','text'])
    accuracy, precision, recall, cm, clf, label_set, count = myModel(dataset)
    return accuracy, precision, recall, cm, clf, label_set, count

# function to render the fprm page so the user can classify arbitrary text
def pred(request):
    return render(request, 'pred.html')

# function that computes the class of the arbitrary user entered text
def result2(request):
    if request.method == 'POST':
        # get the text from the form
        myTest = request.POST['classifyMe']

	    # clean the text
        text = re.sub('[^a-zA-Z0-9]', ' ', myTest)
        myTestList = []
        myTestList.append(text)

	    # load the TfIdf vectorizer and the classification model
        tv = pickle.load(open('tv.sav', 'rb'))
        clf = pickle.load(open('model.sav', 'rb'))

	    # Vectorize the text
        myTest = tv.transform(myTestList).toarray()

	    # make class prediction
        y_pred_test = clf.predict(myTest)
        y_pred_proba = clf.predict_proba(myTest)
        class_proba = y_pred_proba.max()
        y_pred_test = ''.join(y_pred_test)

        return render(request, 'result2.html', {'y_pred_test':y_pred_test, 'y_pred_proba':class_proba})
    else:
        return HttpResponse("Missing form data.  Don't access this URL directly.  Don't use the back button, only hyperlinks and page buttons.")
