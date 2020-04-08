import os
import json
import tensorflow as tf
from tensorflow import keras
import numpy as np
from my_preprocess import preprocessing
from tqdm import tqdm


DIM = 300


def get_model2(embedding_matrix, vocab_size, input_length):
    input1 = keras.layers.Input(shape=(1, ))
    input2 = keras.layers.Input(shape=(1,))
    merged = keras.layers.Concatenate(axis=1)([input1, input2])
    dense1 = keras.layers.Dense(2, input_dim=2, activation=keras.activations.sigmoid, use_bias=True)(merged)
    emb = keras.layers.Embedding(vocab_size,input_length,weights = [embedding_matrix],input_length=input_length,trainable = False)(dense1)
    lstm = keras.layers.Bidirectional(CuDNNLSTM(75))(emb)
    den2 = keras.layers.Dense(32,activation = 'relu')(lstm)
    output = keras.layers.Dense(1,activation = 'sigmoid')(den2)
    model = keras.models.Model(inputs=[input1, input2], output=output)
    

def get_model(embedding_matrix, vocab_size, input_length):
    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size,300,weights = [embedding_matrix],input_length=300,trainable = False))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(75)))
    model.add(keras.layers.Dense(32,activation = 'relu'))
    model.add(keras.layers.Dense(3,activation = 'softmax'))
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics = ['accuracy'])
    return model


def dl_model_train(sen1, sen2, res):
    sen = [sen1[i] + sen2[i] for i in range(len(res))]
    token = keras.preprocessing.text.Tokenizer()
    token.fit_on_texts(sen)
    seq = token.texts_to_sequences(sen)
    MAX_LEN = max([len(i) for i in sen])
    print(MAX_LEN)
    pad_seq = keras.preprocessing.sequence.pad_sequences(seq,maxlen=DIM)
    vocab_size = len(token.word_index)+1
    embedding_vector = {}
    f = open('./glove.6B.300d.txt')
    for line in tqdm(f):
        value = line.split(' ')
        word = value[0]
        coef = np.array(value[1:],dtype = 'float32')
        embedding_vector[word] = coef
    embedding_matrix = np.zeros((vocab_size,DIM))
    for word,i in tqdm(token.word_index.items()):
        embedding_value = embedding_vector.get(word)
        if embedding_value is not None:
            embedding_matrix[i] = embedding_value
    model = get_model(embedding_matrix, vocab_size, MAX_LEN)
    res = np.array(res)
    pad_seq = np.array(pad_seq)
    history = model.fit(pad_seq,res,epochs = 5,batch_size=256,validation_split=0.2)
    
    data_path = os.path.split(os.getcwd())
    data_path = data_path[0] + '/' + data_path[1] + '/snli_1.0'
    with open(os.path.abspath(data_path + '/snli_1.0_test.jsonl'), 'r') as file:
        lines = file.read()
        lines = lines.split('\n')
        nlines = len(lines) - 1
    sen1 = []
    sen2 = []
    res2 = []
    for i in range(nlines):
        x = json.loads(lines[i])
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
            sen1.append(x['sentence1'])
            sen2.append(x['sentence2'])
            res2.append(y)
    sen1 = preprocessing(sen1)
    sen2 = preprocessing(sen2)
    x_test = [sen1[i] + sen2[i] for i in range(len(res2))]
    x_test = token.texts_to_sequences(x_test)
    testing_seq = keras.preprocessing.sequence.pad_sequences(x_test,maxlen=300)
    predict = model.predict_classes(testing_seq)
    sz = len(predict)
    acc = 0
    for i in range(sz):
        if predict[i] == res2[i]:
            acc+=1
    print(acc*100/sz)
    with open('./dl.txt', "w") as file:
        for item in predict:
            if item == 0:
                file.write("contradiction\n")
            elif item == 1:
                file.write("neutral\n")
            elif item == 2:
                file.write("entailment\n")
            else:
                pass
 
    
def dl_model_train2(sen1, sen2, res):
    
    token = keras.Tokenizer()
    token.fit_on_texts(sen1)
    seq1 = token.texts_to_sequences(sen1)
    MAX_LEN = max([len(i) for i in inp_list])
    print(MAX_LEN)
    pad_seq = pad_sequences(seq,maxlen=MAX_LEN)
    vocab_size = len(token.word_index)+1
    embedding_vector = {}
    f = open('./glove.6B.300d.txt')
    for line in tqdm(f):
        value = line.split(' ')
        word = value[0]
        coef = np.array(value[1:],dtype = 'float32')
        embedding_vector[word] = coef
    embedding_matrix = np.zeros((vocab_size,300))
    for word,i in tqdm(token.word_index.items()):
        embedding_value = embedding_vector.get(word)
        if embedding_value is not None:
            embedding_matrix[i] = embedding_value
    model = get_model(embedding_matrix, vocab_size, input_length)
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics = ['accuracy'])
    history = model.fit(pad_seq,res,epochs = 5,batch_size=256,validation_split=0.2)
    
    
    


if __name__ == "__main__":

    
    data_path = os.path.split(os.getcwd())
    data_path = data_path[0] + '/' + data_path[1] + '/snli_1.0'
    with open(os.path.abspath(data_path + '/snli_1.0_train.jsonl'), 'r') as file:
        lines = file.read()
        lines = lines.split('\n')
        nlines = len(lines) - 1
    sen1 = []
    sen2 = []
    res = []
    for i in range(nlines):
        x = json.loads(lines[i])
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
            sen1.append(x['sentence1'])
            sen2.append(x['sentence2'])
            res.append(y)
    
    train_sen1 = preprocessing(sen1)
    train_sen2 = preprocessing(sen2)
    train_res = res
    dl_model_train(train_sen1, train_sen2, train_res)

