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


def content(test, news):
    test_copy = list(test)
    for i in range(len(test_copy)):
        id_str = test_copy[i][1].split(',')
        id_list = [int(i) for i in id_str]
        title_list = []
        content_list = []
        for j in range(len(news)):
            a = news[j]
            content = a['content']
            title = a['title']
            id = a['id']
            if id in id_list:
                title_list.append(title)
                content_list.append(content)
        test_copy[i][1] = content_list
        # test_copy[i].append(title_list)
        # test_copy[i].append(content_list)
    return test_copy


def del_stopword(train):
    train = train.copy()
    for each in train:
        result = []
        for sentence in each[1]:
            # print(sentence)
            cut = jieba.cut(sentence)
            temp = []
            for word in cut:
                if word not in stopWords and judge(word):
                    temp.append(word)
            result.extend(temp)
        each[1] = set(result)
        # print(each[1])
        # break
    return train


def judge(num):
    pattern = re.compile(r'[0-9a-zA-Z\s]')
    result = pattern.match(num)
    if result:
        return False
    else:
        return True



def chi_2(X_test, lable, k=1000):
    tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
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



f = open('news.txt', mode='r', encoding='utf-8')
news = [eval(i) for i in f.readlines()]
f.close()

f = open('train.txt', mode='r', encoding='utf-8')
train = [i.split() for i in f.readlines()]
f.close

f = open('test.txt', mode='r', encoding='utf-8')
test = [i.split() for i in f.readlines()]
f.close

f = open('stopwords.txt', mode='r', encoding='utf-8')
stopWords = set([i for i in f.read() if i != '\n'])
f.close()

new_train = content(train, news)
new_test = content(test, news)

del_new_train = del_stopword(new_train)
del_new_test = del_stopword(new_test)

print('saving data ...')
pickle.dump(del_new_train, open("del_new_train.pkl", "wb"), True)
pickle.dump(del_new_test, open("del_new_test.pkl", "wb"), True)


