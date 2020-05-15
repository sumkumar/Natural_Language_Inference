import numpy as np
import os
import json

import tensorflow as tf
from tensorflow import keras
from my_preprocess import preprocessing
from tfidf_test import lr_test


if __name__ == "__main__":

    

    
    data_path = os.path.split(os.getcwd())
    data_path = data_path[0] + '/' + data_path[1] + '/snli_1.0'
    with open(os.path.abspath(data_path + '/snli_1.0_test.jsonl'), 'r') as file:
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
    
    test_sen1 = preprocessing(sen1)
    test_sen2 = preprocessing(sen2)
    test_res = res
    lr_test(test_sen1, test_sen2, test_res)
    
    
    