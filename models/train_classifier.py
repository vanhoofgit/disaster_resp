#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 12:16:48 2020

@author: benedictusvanhoof
"""

import sys
from sqlalchemy import create_engine
import numpy as np
import pandas as pd
import nltk
import pickle
nltk.download('stopwords')
from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.metrics import confusion_matrix,accuracy_score, precision_score, recall_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline






def load_data(database_filepath):
    '''
    This function loads the data from an sqllite database into a    
    dataframe df
    input : the path to the sqllite database
    output : a pandas dataframe
    '''
    
    database_filepath = 'sqlite:///' + database_filepath
    engine = create_engine(database_filepath)            
    df = pd.read_sql("SELECT * FROM DisasterMessages", engine)
    X = df.message.values
    y = df.iloc[:,4:41].values
    category_names = df.columns[4:41]    
    return X, y , category_names



def tokenize(text):
    ''' this function puts all the words into lowercase, splits the text into tokens and gets rid of    
    the punctuations and stopwords
    '''
    
    # normlize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
   
    # split into tokens
    tokens = word_tokenize(text)
    
    # get rid of the stopwords
    meaningful_words = [w for w in tokens if w not in stopwords.words("english")]
    
    # lemmatization
    
    lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in meaningful_words]
    
    return lemmed
             

def build_model( X,Y):
    '''
    This function builds the classifier model (Random Forest) with Pipeline and GridSearch)
    As input it takes the numpay arrays with the features and the labe
    
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf',MultiOutputClassifier(RandomForestClassifier(n_jobs = -1))),])
    
    parameters = {'clf__estimator__criterion' : ['entropy'],
              'clf__estimator__max_depth': [2,None],
              'clf__estimator__n_estimators':[10,100,]}

    limited_parameters = {'clf__estimator__n_estimators':[10,100],
                     'vect__ngram_range': [(1,1)]}

    cv = GridSearchCV(estimator=pipeline, param_grid=limited_parameters, scoring = 'f1_micro', cv=5, n_jobs = 1, verbose = 10)     

    ## train the model with gridsearch
    cv.fit(X_train,y_train)
    # make a prediction
    predicted = cv.predict(X_test)
    ## print out the best parameters
    best_parameters = cv.best_params_
   
    print (best_parameters)
    
    ## print out the best score   
    best_score = cv.best_score_
    print(best_score)  
    
    return cv,y_test,predicted
                    
   




def evaluate_model(y_test,predicted):
    '''
    This function calculates the accuracy, precision, recall
    The input is the true test labels and the predicted labels
    The function doesn't return anything
    
    
    '''
    
    ## Calculate the accuracy score as a proportion
    ac_sc = accuracy_score(y_test, predicted,normalize=True, sample_weight=None)
    print ("Accuracy score :", ac_sc) 
    ## Calculate the precision score
    pr_sc = precision_score(y_test, predicted,labels=None, average='micro', sample_weight=None, zero_division='warn')
   
    print ("Precision score :", pr_sc)
    ## Calculate the recall score
    re_sc = recall_score(y_test, predicted,labels=None, average='micro', sample_weight=None, zero_division='warn')
    print ("Recall score :", re_sc) 
   


    
    


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
       
        
        print('Building model...')
        model, y_test, predicted = build_model(X,Y)        
        
        print('Evaluating model...')  
        evaluate_model( y_test, predicted)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()