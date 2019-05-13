import nltk
import jieba
import heapq
import re
import pickle
import copy
import numpy as np
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


def score(result, test):
    f = open(result, 'r')
    predictions = [i.strip().split() for i in f.readlines()]
    # print(predictions)
    f.close()

    n = len(predictions)
    TP = len([i for i in range(n) if predictions[i][0]
              == '+1' and predictions[i][1] == '+1'])
    FN = len([i for i in range(n) if predictions[i][0]
              == '-1' and predictions[i][1] == '+1'])
    FP = len([i for i in range(n) if predictions[i][0]
              == '+1' and predictions[i][1] == '-1'])
    TN = len([i for i in range(n) if predictions[i][0]
              == '-1' and predictions[i][1] == '-1'])

    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    accuracy = (TP + TN) / n
    error_rate = 1 - accuracy
    F1 = 2 * precision * recall / (precision + recall)
    return recall, precision, accuracy, error_rate, F1

def feature_gen(train_in,test_in):
    tfidf = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b",
        # max_df=0.7, min_df=0.1,
    )
    train_in = tfidf.fit_transform(train_in).toarray()

    pickle.dump(tfidf.vocabulary, open("VOCAB.pkl", "wb"), True)

    tfidf2 = TfidfVectorizer(vocabulary=tfidf.vocabulary_)
    test_in = tfidf2.fit_transform(test_in)
    return train_in,test_in


print('loading data ...')
del_new_train = pickle.load(open('del_new_train.pkl', 'rb'))
del_new_test = pickle.load(open('del_new_test.pkl', 'rb'))

pos_train_text = [' '.join(each[1])
                  for each in del_new_train if each[0] == '+1']
neg_train_text = [' '.join(each[1])
                  for each in del_new_train if each[0] == '-1']

print('data processing ...')

train_in = pos_train_text + neg_train_text
test_in = [' '.join(each[1]) for each in del_new_test]
train_label = [1] * len(pos_train_text) + [0] * len(neg_train_text)
test_label = [each[0] for each in del_new_test]


train_feature, test_feature = feature_gen(train_in,test_in)

print('building model ...')
logreg = linear_model.LogisticRegression(C=1,
                                         # penalty='l1',
                                         # solver='liblinear',
                                         penalty='l2',
                                         solver = 'sag',
                                         )
logreg.fit(train_feature, train_label)


# print(test_feature.shape, train_feature.shape, test_in.shape, train_in.shape)
# print(type(test_feature),len(test_feature),len(test_feature[0]),len(train_feature),len(train_feature[0]))
# (3000, 1882) (7954, 1882) (3000, 1882) (7954, 1882)
pred = logreg.predict(test_feature)
pickle.dump(logreg, open("LR.pkl", "wb"), True)

PRED = []
for each in pred:
    if each == 1:
        PRED.append('+1')
    else:
        PRED.append('-1')

# print(PRED)
print('writing result into file ...')
f = open('result_lr', 'w')
for i in range(len(pred)):
    f.write(str(PRED[i]) + ' ' + str(test_label[i]) + '\n')
f.close()

print('evaling ...')
print(score('result_lr', 'test.txt'))
# (0.856544502617801, 0.7051724137931035, 0.6806666666666666, 0.31933333333333336, 0.7735224586288416)
