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
    TFIDF_vect = TfidfVectorizer()
    TFIDF_vect.fit(corpus)
    with open('./model/TFIDF.pickle', "wb") as file:
        pickle.dump(TFIDF_vect, file)
    tfidf_sen1 = TFIDF_vect.transform(corpus_sen1)
    tfidf_sen2 = TFIDF_vect.transform(corpus_sen2)
    feature_array = scipy.sparse.hstack((tfidf_sen1, tfidf_sen2))
    return feature_array
    

def lr_train(sen1, sen2, res):
    train_feature = get_features(sen1, sen2)
    LR_model = LogisticRegression(random_state=0, max_iter=2500, solver='lbfgs', multi_class='auto')
    LR_model.fit(train_feature, res)

    with open('./model/LR.pickle', "wb") as file:
        pickle.dump(LR_model, file)
    


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
    lr_train(train_sen1, train_sen2, train_res)