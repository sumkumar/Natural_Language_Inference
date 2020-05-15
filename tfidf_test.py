import pickle
import numpy as np
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def get_features(sen1, sen2):
    num_samples = len(sen1)
    corpus_sen1 = [' '.join(item) for item in sen1]
    corpus_sen2 = [' '.join(item) for item in sen2]
    corpus = [corpus_sen1[i] + " " + corpus_sen2[i] for i in range(num_samples)]
    with open('./model/TFIDF.pickle', "rb") as file:
        TFIDF_vect = pickle.load(file)
    tfidf_sen1 = TFIDF_vect.transform(corpus_sen1)
    tfidf_sen2 = TFIDF_vect.transform(corpus_sen2)
    feature_array = scipy.sparse.hstack((tfidf_sen1, tfidf_sen2))
    return feature_array


def lr_test(sen1, sen2, res):
    test_feature = get_features(sen1, sen2)

    with open('./model/LR.pickle', "rb") as file:
        LR_model = pickle.load(file)

    pred_labels = LR_model.predict(test_feature)

    with open('./tfidf.txt', "w") as file:
        for item in pred_labels:
            if item == 0:
                file.write("contradiction\n")
            elif item == 1:
                file.write("neutral\n")
            elif item == 2:
                file.write("entailment\n")
            else:
                pass

    # Evaluate and print the results
    score = LR_model.score(test_feature, res) * 100
    print("Accuracy score using Logistic Regression on TF-IDF features is {:.2f}%.".format(score))