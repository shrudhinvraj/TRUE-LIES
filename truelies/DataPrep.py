# -*- coding: utf-8 -*-

import os
import pandas as pd
import csv
import numpy as np
import nltk
from nltk.stem import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import seaborn as sb

#before reading the files, setup the working directory to point to project repo
#reading data files 


test_filename = 'test.csv'
train_filename = 'train.csv'
valid_filename = 'valid.csv'

train_ad  = pd.read_csv(train_filename)
test_ad = pd.read_csv(test_filename)
valid_ad = pd.read_csv(valid_filename)



#data observation
def data_obs():
    print("training dataset size:")
    print(train_ad.shape)
    print(train_ad.head(10))

    #below dataset were used for testing and validation purposes
    print(test_ad.shape)
    print(test_ad.head(10))
    
    print(valid_ad.shape)
    print(valid_ad.head(10))

#check the data by calling below function
#data_obs()

#distribution of classes for prediction
def create_distribution(dataFile):
    
   
    return sb.countplot(x='Label', data=dataFile, palette='hls')

#by calling below we can see that training, test and valid data seems to be failry evenly distributed between the classes
create_distribution(train_ad)
create_distribution(test_ad)
create_distribution(valid_ad)


#data integrity check (missing label values)
#none of the datasets contains missing values therefore no cleaning required
def data_qualityCheck():
    
    print("Checking data qualitites...")
    train_ad.isnull().sum()
    train_ad.info()
        
    print("check finished.")

    #below datasets were used to 
    test_ad.isnull().sum()
    test_ad.info()

    valid_ad.isnull().sum()
    valid_ad.info()

#run the below function call to see the quality check results
data_qualityCheck()



eng_stemmer = SnowballStemmer('english')
stopwords = set(nltk.corpus.stopwords.words('english'))

#Stemming
def stem_tokens(tokens, stemmer):
    stemmed = []
    for token in tokens:
        stemmed.append(stemmer.stem(token))
    return stemmed

#process the data
def process_data(data,exclude_stopword=True,stem=True):
    tokens = [w.lower() for w in data]
    tokens_stemmed = tokens
    tokens_stemmed = stem_tokens(tokens, eng_stemmer)
    tokens_stemmed = [w for w in tokens_stemmed if w not in stopwords ]
    return tokens_stemmed
process_data(train_ad)

#creating ngrams
#unigram 
def create_unigram(words):
    assert type(words) == list
    return words



porter = PorterStemmer()

def tokenizer(text):
    return text.split()


def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


 
