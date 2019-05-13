import nltk
import jieba
import heapq
import re
import pickle
import copy
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


def chi_2(X_test, lable, k=1000):
    tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b",max_df=0.7, min_df=0.1,)
    weight = tfidf.fit_transform(X_test).toarray()
    word = tfidf.get_feature_names()
    ch2 = SelectKBest(chi2, k=k)
    x = ch2.fit_transform(weight, lable)
    word = np.array(word)[ch2.get_support(indices=True)]
    return list(word)


def prepare(train,vocab):
    train_sample = []
    pattern = dict()
    for each in vocab:
        pattern[each] = False
    for each in train:
        words = each[1]
        pattern1 = copy.deepcopy(pattern)
        for word in words:
            if word in vocab:
                pattern1[word] = True
        train_sample.append((pattern1, each[0]))
    return train_sample


def score(result,test):
    f = open(result,'r')
    predictions = [i.split() for i in f.readlines()]
    # print(predictions)
    f.close()

    n = len(predictions)
    TP = len([i for i in range(n) if predictions[i][1] == '+1' and predictions[i][0] == '+1'])
    FP = len([i for i in range(n) if predictions[i][1] == '-1' and predictions[i][0] == '+1'])
    FN = len([i for i in range(n) if predictions[i][1] == '+1' and predictions[i][0] == '-1'])
    TN = len([i for i in range(n) if predictions[i][1] == '-1' and predictions[i][0] == '-1'])

    recall = TP/(TP+FN)
    precision = TP/(TP+FP)
    accuracy = (TP+TN)/n
    error_rate = 1-accuracy
    F1 = 2 * precision * recall/(precision+recall)
    return recall,precision,accuracy,error_rate,F1


print('loading data ...')
del_new_train = pickle.load(open('del_new_train.pkl', 'rb'))
del_new_test = pickle.load(open('del_new_test.pkl', 'rb'))

pos_train_text = [' '.join(each[1])
                  for each in del_new_train if each[0] == '+1']
neg_train_text = [' '.join(each[1])
                  for each in del_new_train if each[0] == '-1']


print('chi2 process ...')
VOCAB = []
K = list(range(50,1851,50))
EVAL = []
for k in K:
    vocab = chi_2(pos_train_text+neg_train_text,[1]*len(pos_train_text)+[0]*len(neg_train_text),k)
    VOCAB.append(vocab)

print('length of vocab:', len(VOCAB))
for j,vocab in enumerate(VOCAB):
    prepare_del_new_train = prepare(del_new_train,vocab)
    prepare_del_new_test = prepare(del_new_test,vocab)

    classifier = nltk.NaiveBayesClassifier.train(prepare_del_new_train)  # 生成分类器
    print(j,'writing result into file ...')
    f = open('result_v2_'+str(50+j*50), 'w')
    for i in range(len(prepare_del_new_test)):
        tag = classifier.classify(prepare_del_new_test[i][0])  # 分类
        f.write(str(tag) + ' ' + str(prepare_del_new_test[i][1]) + '\n')
    f.close()
    print(j,'evaling ...')
    
    EVAL.append(score('result_v2_'+str(50+j*50),'test.txt'))
    print(j,'done ...')

pickle.dump({'EVAL':EVAL,'K':K}, open("result_v2.pkl", "wb"), True)
print('all done')


