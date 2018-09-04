# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 11:55:19 2018

@author: Usman
"""

# It was really a struggle trying to get the textblob libraries installed.

# Import libraries
import csv
from textblob import TextBlob
import pandas
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

# Load the training dataset 'SMSpanCollection' into variable 'messages'
messages = [line.strip() for line in open('SMSSpamCollection')]

# print number of messages
print(len(messages))

"""
Read the dataset. Specify the field separator is a tab instead of a comma.
Additionally, add column captions ('label' and 'message') for the two 
fields in the dataset.
To preserve internal quotations in messages, use QUOTE_NONE.        

"""
messages = pandas.read_csv('SMSSpamCollection', sep='\t',
                           quoting=csv.QUOTE_NONE,
                           names=["class", "message"])

# print first five records
print(messages.head())

# groupby class and count
print(messages.groupby('class').count())

# split messages into individual words
def SplitIntoWords(messages):
    return TextBlob(messages).words

# this is what the first 5 records look when splitted into individual words
print(messages.message.head().apply(SplitIntoWords))

# convert each word into its base form
def WordsIntoBaseForm(message):
    words = TextBlob(message.lower()).words
    return [word.lemma for word in words]

# convert each word into unique vector
trainingVector = CountVectorizer(analyzer=WordsIntoBaseForm).fit(messages['message'])

# view occurenct of words in an arbitrary vector
message10 = trainingVector.transform([messages['message'][9]])
print(message10)

# print message10 for comparism
print(messages['message'][9])

# identify repeated words
print('First word that appears twice: ', trainingVector.get_feature_names()[3433])
print('First word that appears trice: ', trainingVector.get_feature_names()[5182])

# bag of words for the entire training set
messagesBagofWords = trainingVector.transform(messages['message'])

# weight of words in the entire training dataset - Term Frequecy and Inverse Document Frequency
messagesTfidf = TfidfTransformer().fit(messagesBagofWords).transform(messagesBagofWords)

# train the model using naive bayes algorithm
spamDetector = MultinomialNB().fit(messagesTfidf, messages['class'].values)