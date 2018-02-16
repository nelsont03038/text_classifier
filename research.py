#####################################
### Getting and cleaning the data ###
#####################################

# Importing numpy and pandas packages
import numpy as np
import pandas as pd

# Importing the dataset as a pandas dataframe
dataset = pd.read_table('spooky_authors_train.txt', sep='\t', header=None, names=['label','text'])

# Cleaning the text
# remove non-alphanumeric characters
# convert to lower case
# saving the clean text corpus as a list of lists
# I tried word stemming and word lemmatizing but it 
# did not improve model performance
import re
corpus = []
for i in range(len(dataset)):
    text = re.sub('[^a-zA-Z0-9]', ' ', dataset['text'][i])
    corpus.append(text)
del i, text

# format data as numpy arrays
X = np.array(corpus)
y = dataset.iloc[:, 0].values

# Splitting the dataset into the training set and test set
# 80% for training, 20% for testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

##########################################
### Research on models and performance ###
##########################################

## Multinomial Naive Bayes Classifier ##

# Build a pipeline - we use a pipeline so we can study
# various parameters across a grid search on the various
# steps.  The steps here include removing stopwords, 
# converting text into numerical vectors, transforming 
# the count vector space into term frequency and/or 
# inverse document frequency space, then using the 
# naive Bayes method for classification
# the grid search uses k-fold cross validation for accurate
# measure of bias and variance, then chooses the optimal parameters
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
classifier = Pipeline([('vect', CountVectorizer(stop_words='english')),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB()),
])

# parameter tuning with grid search
from sklearn.model_selection import GridSearchCV
parameters = {'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
              'tfidf__norm': ['l1', 'l2', None],
              'tfidf__use_idf': (True, False),
              'clf__alpha': (0.1, 0.5, 1.0),
}
clf = GridSearchCV(classifier, parameters, cv=3)
clf.fit(X_train, y_train) # takes 5 or 10 minutes

# Predicting the Test set results
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

# retrieve the best paramters as determined by gridsearch
# over hyperparameter space and k-fold cross validation
best_parameters = clf.best_params_

# write summary to console
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("Confusion matrix\n", cm)
print("The best parameters are: ", best_parameters)

"""
With the larger data set, I boostrap sampled different 
train/test splits and get the same parameters each time.
With the small data set, it is too small to converge on 
one best parameter space and we will have high bias and 
variance as a result.  Expected from small data sets.

~84% accuracy

Best parameters

'vect__ngram_range': (1, 3)
'tfidf__norm': None
'tfidf__use_idf': False
'clf__alpha': 0.5
"""

### Final model for web application ###

# Creating the TfIdf model (convert text into numerical vectors)
from sklearn.feature_extraction.text import TfidfVectorizer
tv = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), use_idf=False, norm=None)
X = tv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 0].values
   
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

# creating a simple multinomial naive bayes classifier
# and fitting the model
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB(alpha=0.1)
clf.fit(X_train, y_train)
    
# Predicting the test set results
y_pred = clf.predict(X_test)

# Measuring model performance
# Making the Confusion Matrix and model performance measures
# accuracy - true positives + true negatives/ total
# precision - true positives / (true positives+false positives)
# recall - true positives / (true positives + false negatives)
# f1 - harmonic mean of precision and recall
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')             
recall = recall_score(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred)





## Random Forest Classifier ##

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
classifier = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', RandomForestClassifier()),
])

# parameter tuning with grid search
from sklearn.model_selection import GridSearchCV
parameters = {'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
              'tfidf__use_idf': (True, False),
              'clf__n_estimators': [10, 50, 100],
              'clf__criterion': ['gini', 'entropy']
}
clf = GridSearchCV(classifier, parameters, cv=3)
clf.fit(X_train, y_train) # CAUTION - this will take a long time
# maybe as long as 2 hours

# Predicting the Test set results
y_pred = clf.predict(X_test)

# Measuring model performance
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')             
recall = recall_score(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred)

# retrieve the best paramters as determined by gridsearch
best_parameters = clf.best_params_

# write summary to console
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("Confusion matrix\n", cm)
print("The best parameters are: ", best_parameters)

"""
~73% accuracy
Best parameters
'clf__n_estimators': 100
'tfidf__use_idf': True
'vect__ngram_range': (1, 2)
'clf__criterion': 'gini'
"""





## Support Vector Machines Classifier ##

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
classifier = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', SVC()),
])

# parameter tuning with grid search
from sklearn.model_selection import GridSearchCV
parameters = {'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
              'tfidf__use_idf': (True, False),
              'clf__C': [0.3, 0.5, 1.0, 1.3],
              'clf__kernel': ['linear', 'rbf']
}
clf = GridSearchCV(classifier, parameters, cv=3)
clf.fit(X_train, y_train) # CAUTION - this will take a long time too

# Predicting the Test set results
y_pred = clf.predict(X_test)

# Measuring model performance
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')             
recall = recall_score(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred)

# retrieve the best paramters as determined by gridsearch
best_parameters = clf.best_params_

# write summary to console
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("Confusion matrix\n", cm)
print("The best parameters are: ", best_parameters)

"""
~82% accuracy
Best parameters
tfidf__use_idf': True
'clf__kernel': 'linear'
'vect__ngram_range': (1, 2)
'clf__C': 1.3
"""



