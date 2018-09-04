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