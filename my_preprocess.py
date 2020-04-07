import re
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk.stem import PorterStemmer
ps = PorterStemmer()
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
#import num2word can be used if numbers are important
from collections import Counter 
def remove_duplicate_words(list):
    return Counter(list).keys()
def preprocessing(inp_list):
    
    new_list = []
    for doc in inp_list:
        doc = doc.lower()
        doc = re.sub(r'\d+', '', doc)
        doc = doc.translate(str.maketrans("","", string.punctuation))
        doc = doc.strip()
        tokens = nltk.word_tokenize(doc)
        tokens = [i for i in tokens if not i in stop_words]
        tokens = [ps.stem(i) for i in tokens]
        tokens = remove_duplicate_words(tokens)
        tokens = [lemmatizer.lemmatize(i) for i in tokens]
        tokens = [i for i in tokens if len(i) > 1]
        tokens = remove_duplicate_words(tokens)
        new_list.append(list(tokens))
    return new_list