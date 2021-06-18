# -*- coding: utf-8 -*-
"""
Created on Sun May 30 21:38:44 2021

@author: DELL
"""
#Perform sentimental analysis on the Elon-musk tweets (Exlon-musk.csv)
#Importing the libraries
import pandas as pd
import numpy as np
from matplotlib.pyplot import imread
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import spacy
import string

#loading the dataset
tweet = pd.read_csv('Elon_musk.csv',encoding='latin1')
tweet.head(20)
tweet.columns
tweet = tweet.drop(['Unnamed: 0'], axis=1)
tweet.head(10)

#Data cleaning steps
#Removing the unnecessary punctuation tags
tweet = [Text.strip() for Text in tweet.Text]
tweet = [Text for Text in tweet if Text]
tweet[0:10]
text = ''.join(tweet)
len(text)

no_punc_text = text.translate(str.maketrans('','', string.punctuation))
no_punc_text

import nltk
nltk.download('punkt')

#Tokenization(split the text into words)
from nltk.tokenize import word_tokenize
text_tokens = word_tokenize(no_punc_text)
print(text_tokens[0:50])
len(text_tokens)

#Removing the stopwords
from nltk.corpus import stopwords
nltk.download('stopwords')
my_stop_words = stopwords.words('english')
my_stop_words.append('the')
no_stop_tokens= [word for word in text_tokens if not word in my_stop_words]
print(no_stop_tokens[0:50])

#lowering the cases (caps to small letter)
lower_words=[Text.lower() for Text in no_stop_tokens]
print(lower_words[0:50])

#Stemming
from nltk.stem import PorterStemmer
stemmed_tokens=[PorterStemmer().stem(word) for word in lower_words]
print(stemmed_tokens[0:50])

from spacy.lang.en.examples import sentences
nlp = spacy.load('en_core_web_sm')
doc = nlp(''.join(no_stop_tokens))
print(doc[0:50])

#Lemmatization
from nltk.stem import WordNetLemmatizer
lemmas = [token.lemma_ for token in doc]
print(lemmas[0:50])

#Feature extraction(TF-IDF and Bag of words)
#TF-IDF
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(lemmas)
x
print(vectorizer.get_feature_names()[0:50])
print(x.toarray())

#Generate the word cloud(Bag of words)
%matplotlib inline
from wordcloud import WordCloud, STOPWORDS
def word_cloud(wordcloud):
    plt.figure(figsize = (40,30))
    plt.imshow(wordcloud)
    plt.axis('off');
    
stopwords = STOPWORDS
stopwords.add('will')    
wordcloud = WordCloud(width=3000, height=2000, background_color='black', max_words=100, stopwords=stopwords).generate(text)
word_cloud(wordcloud)














