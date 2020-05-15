# -*- coding: utf-8 -*-
"""Assignment3_DL_model_new.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1O9k3eMmmn45jBC5MsEh0b1lXO9PCn1HZ
"""

import nltk
!python -m nltk.downloader punkt
!python -m nltk.downloader stopwords
!python -m nltk.downloader wordnet

import os
import json
import tensorflow as tf
from tensorflow import keras
import numpy as np
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
from tqdm import tqdm



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

def get_model(embedding_matrix, vocab_size, input_length):
    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size,300,weights = [embedding_matrix],input_length=300,trainable = False))
    #model.add(keras.layers.Bidirectional(keras.layers.LSTM(1024, return_sequences=True)))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(512, return_sequences=True)))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True)))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True)))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True)))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(128)))
    model.add(keras.layers.Dense(4096,activation = 'relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(1024,activation = 'relu'))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Dense(32,activation = 'relu'))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(3,activation = 'softmax'))
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics = ['accuracy'])
    return model

def lstm_test():
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        MLP_model_name = 'models/MLP_model_CPU.h5'
        #convnet_model_name = 'models/convnet_model_CPU.h5'
        #raise SystemError('GPU device not found')
    else :
        MLP_model_name = 'models/MLP_model_GPU.h5'
        #convnet_model_name = 'models/convnet_model_GPU.h5'
    print('Found GPU at: {}'.format(device_name))
    sen1_test = []
    sen2_test = []
    res_test = []
    sen1_train = []
    sen2_train = []
    res_train = []
    resp = urlopen("https://nlp.stanford.edu/projects/snli/snli_1.0.zip")
    zipfile = ZipFile(BytesIO(resp.read()))
    for line in zipfile.open('snli_1.0/snli_1.0_train.jsonl').readlines():
        x = json.loads(line)
        k = 0
        y = -1
        if x['gold_label'] == "contradiction":
            y = 0
            k = 1
        elif x['gold_label'] == "neutral":
            y = 1
            k = 1
        elif x['gold_label'] == "entailment":
            y = 2
            k = 1
        if k==1:
            sen1_train.append(x['sentence1'])
            sen2_train.append(x['sentence2'])
            res_train.append(y)
    sen1_train = preprocessing(sen1_train)
    sen2_train = preprocessing(sen2_train)
    for line in zipfile.open('snli_1.0/snli_1.0_test.jsonl').readlines():
        x = json.loads(line)
        k = 0
        y = -1
        if x['gold_label'] == "contradiction":
            y = 0
            k = 1
        elif x['gold_label'] == "neutral":
            y = 1
            k = 1
        elif x['gold_label'] == "entailment":
            y = 2
            k = 1
        if k==1:
            sen1_test.append(x['sentence1'])
            sen2_test.append(x['sentence2'])
            res_test.append(y)
    sen1_test = preprocessing(sen1_test)
    sen2_test = preprocessing(sen2_test)

    encoding = 'utf-8'
    resp = urlopen("http://nlp.stanford.edu/data/glove.840B.300d.zip")
    zipfile = ZipFile(BytesIO(resp.read()))
    embedding_vector = {}
    for line in zipfile.open('glove.840B.300d.txt').readlines():
        line = line.decode(encoding)
        value = line.split(' ')
        word = value[0]
        coef = np.array(value[1:],dtype = 'float32')
        embedding_vector[word] = coef

    DIM = 300
    sen = [sen1[i] + sen2[i] for i in range(len(res))]
    token = keras.preprocessing.text.Tokenizer()
    token.fit_on_texts(sen)
    seq = token.texts_to_sequences(sen)
    MAX_LEN = max([len(i) for i in sen])
    print(MAX_LEN)
    pad_seq = keras.preprocessing.sequence.pad_sequences(seq,maxlen=DIM)
    vocab_size = len(token.word_index)+1
    embedding_matrix = np.zeros((vocab_size,DIM))
    for word,i in tqdm(token.word_index.items()):
        embedding_value = embedding_vector.get(word)
        if embedding_value is not None:
            embedding_matrix[i] = embedding_value
    model = get_model(embedding_matrix, vocab_size, MAX_LEN)
    res = np.array(res)
    pad_seq = np.array(pad_seq)
    model.load_weights('./model/dl_765.h5')
    x_test = [sen1_test[i] + sen2_test[i] for i in range(len(res_test))]
    x_test = token.texts_to_sequences(x_test)
    testing_seq = keras.preprocessing.sequence.pad_sequences(x_test,maxlen=300)
    predict = model.predict_classes(testing_seq)
    sz = len(predict)
    acc = 0
    for i in range(sz):
        if predict[i] == res_test[i]:
            acc+=1
    print(acc*100/sz)
    with open('./deep_model_lstm.txt', "w") as file:
        for item in predict:
            if item == 0:
                file.write("contradiction\n")
            elif item == 1:
                file.write("neutral\n")
            elif item == 2:
                file.write("entailment\n")
            else:
                pass