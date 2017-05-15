# -*- coding:utf-8 -*-
# =======================================================
# 
# @FileName  : first_predict.py
# @Author    : Wang Hongqing
# @Date      : 2017-05-13 20:40
# 
# =======================================================

import logging
import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.cross_validation import train_test_split
from collections import Counter
from nltk.corpus import stopwords

reload(sys)
sys.setdefaultencoding('utf-8')

logging.basicConfig(
    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
    level=logging.DEBUG,
    datefmt='%a, %d %b %Y %H:%M:%S'
)


def word_match_share(row):
    """
    计算相同的词的个数, 占两句话总次数的比例
    :param row: 
    :return: 
    """
    # TODO 这个函数需要进行优化
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # 如果两句话恰好全是停止词
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    ratio = (len(shared_words_in_q1) + len(shared_words_in_q2)) / (len(q1words) + len(q2words))
    return ratio


def get_weight(count, eps=10000, min_count=2):
    """
    计算每个词在所给的quora语料库之中的权重
    如果一个词在语料库之中只出现一次， 那么久忽略这个词
    
    :param count: 
    :param eps: 
    :param min_count: 
    :return: 
    """
    # TODO 定义一个平滑常量， 让极端少数出现的词造成的影响尽可能的小
    if count < min_count:
        return 0
    else:
        return 1.0 / (count + eps)


def tfidf_word_match_share(row):
    """
    计算每个词的tf_idf结果
    :param row: 
    :return: 
    """
    q1_set = set()
    q2_set = set()
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1_set.add(word)
    
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2_set.add(word)

    if len(q1_set) == 0 and len(q2_set) == 0:
        # 两句话都是只有停用词
        return 0

    shared_weights = [weights.get(w, 0) for w in q1_set & q2_set]
    total_weights = [weights.get(w, 0) for w in q1_set] + [weights.get(w, 0) for w in q1_set]
    
    ratio = np.sum(shared_weights) / np.sum(total_weights)
    return ratio


home_dir = os.getcwd()
data_dir = os.path.join(home_dir, "data")
result_dir = os.path.join(home_dir, "result")

train_file = os.path.join(data_dir, "train.csv")
test_file = os.path.join(data_dir, "test.csv")
result_file = os.path.join(result_dir, "navie_submission.csv")

train = pd.read_csv(train_file)
test = pd.read_csv(test_file)

train_qs = pd.Series(train['question1'].tolist() + train['question2'].tolist()).astype(str)
test_qs = pd.Series(test['question1'].tolist() + test['question2'].tolist()).astype(str)

# 英语之中的无用的停止词
stops = set(stopwords.words("english"))
eps = 5000
words = (" ".join(train_qs)).lower().split()
counts = Counter(words)
weights = {word: get_weight(count) for word, count in counts.items()}

train_word_match = train.apply(word_match_share, axis=1, raw=True)
tfidf_train_word_match = train.apply(tfidf_word_match_share, axis=1, raw=True)

# First we create our training and testing data
x_train = pd.DataFrame()
x_test = pd.DataFrame()
x_train['word_match'] = train_word_match
x_train['tfidf_word_match'] = tfidf_train_word_match
x_test['word_match'] = test.apply(word_match_share, axis=1, raw=True)
x_test['tfidf_word_match'] = test.apply(tfidf_word_match_share, axis=1, raw=True)

y_train = train['is_duplicate'].values

pos_train = x_train[y_train == 1]
neg_train = x_train[y_train == 0]

# 正采样过多，负采样过少， 这样的话， 需要对负样本进行过采样
# TODO 一定要知道p值是如何确定的。
p = 0.165
scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
while scale > 1:
    neg_train = pd.concat([neg_train, neg_train])
    scale -= 1
neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])

x_train = pd.concat([pos_train, neg_train])
y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()
del pos_train
del neg_train

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)

# 设置xgboost参数
params = dict()
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.02
params['max_depth'] = 4

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)

sub = pd.DataFrame({'test_id': test['test_id'], 'is_duplicate': p})
sub.to_csv(result_file, index=False)
