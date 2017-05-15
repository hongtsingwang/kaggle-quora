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
from collections import Counter

import numpy as np
import pandas as pd
import xgboost as xgb

from __future__ import division
from nltk.corpus import stopwords
from sklearn.cross_validation import train_test_split

reload(sys)
sys.setdefaultencoding('utf-8')

logging.basicConfig(
    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
    level=logging.DEBUG,
    datefmt='%a, %d %b %Y %H:%M:%S'
)

logging.info("the program start!")
logging.info(u"这次目标是优化xgboost")
random_seed = 12357  # 尝试过的其他参数4242
xgboost_iterations = 315


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


def word_share(row):
    """
    这个函数没有搞太懂
    :param row: 
    :return: 
    """
    q1_list = str(row['question1']).lower().split()
    q1 = set(q1_list)
    q1words = q1.difference(stops)
    if len(q1words) == 0:
        return '0:0:0:0:0:0:0:0'

    q2_list = str(row['question2']).lower().split()
    q2 = set(q2_list)
    q2words = q2.difference(stops)

    if len(q2words) == 0:
        return '0:0:0:0:0:0:0:0'

    # 对这句话有疑问
    words_hamming = sum(1 for i in zip(q1_list, q2_list) if i[0] == i[1]) / max(len(q1_list), len(q2_list))

    q1stops = q1.intersection(stops)
    q2stops = q2.intersection(stops)

    q1_2gram = set([i for i in zip(q1_list, q1_list[1:])])
    q2_2gram = set([i for i in zip(q2_list, q2_list[1:])])

    shared_2gram = q1_2gram.intersection(q2_2gram)

    shared_words = q1words.intersection(q2words)
    shared_weights = [weights.get(w, 0) for w in shared_words]
    q1_weights = [weights.get(w, 0) for w in q1words]
    q2_weights = [weights.get(w, 0) for w in q2words]
    total_weights = q1_weights + q1_weights

    R1 = np.sum(shared_weights) / np.sum(total_weights)  # tfidf share
    R2 = len(shared_words) / (len(q1words) + len(q2words) - len(shared_words))  # count share
    R31 = len(q1stops) / len(q1words)  # stops in q1
    R32 = len(q2stops) / len(q2words)  # stops in q2
    Rcosine_denominator = (np.sqrt(np.dot(q1_weights, q1_weights)) * np.sqrt(np.dot(q2_weights, q2_weights)))
    Rcosine = np.dot(shared_weights, shared_weights) / Rcosine_denominator
    if len(q1_2gram) + len(q2_2gram) == 0:
        R2gram = 0
    else:
        R2gram = len(shared_2gram) / (len(q1_2gram) + len(q2_2gram))
    return '{}:{}:{}:{}:{}:{}:{}:{}'.format(R1, R2, len(shared_words), R31, R32, R2gram, Rcosine, words_hamming)


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
    if len(q1_set) == 0 or len(q2_set) == 0:
        # 两句话都是只有停用词
        return 0

    shared_weights = [weights.get(w, 0) for w in q1_set & q2_set]
    total_weights = [weights.get(w, 0) for w in q1_set] + [weights.get(w, 0) for w in q1_set]
    ratio = sum(shared_weights) * 1.0 / sum(total_weights)
    return ratio


def train_xgb(X, y, params):
    """
    用xgboost对数据进行训练
    :param X: 训练集
    :param y: 标签
    :param params: xgboost 参数
    :return: 
    """
    logging.info("we will train xgboost for %d iterations, set random seed %d" % (xgboost_iterations, random_seed))
    x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=random_seed)

    xgb_train = xgb.DMatrix(x_train, label=y_train)
    # 验证集
    xgb_valid = xgb.DMatrix(x_valid, label=y_valid)
    watch_list = [(xgb_train, "train"), (xgb_valid, "eval")]
    return xgb.train(params, xgb_train, xgboost_iterations, watch_list)


def predict_xgb(clr, X_test):
    return clr.predict(xgb.DMatrix(X_test))


def add_word_count(x, df, word):
    """
    这个函数看着好奇怪的样子
    :param x: 
    :param df: 
    :param word: 
    :return: 
    """
    x['q1_' + word] = df['question1'].apply(lambda x: (word in str(x).lower()) * 1)
    x['q2_' + word] = df['question2'].apply(lambda x: (word in str(x).lower()) * 1)
    x[word + '_both'] = x['q1_' + word] * x['q2_' + word]


def main():
    # 设置xgboost参数
    params = dict()
    params['objective'] = 'binary:logistic'
    params['eval_metric'] = 'logloss'
    params['eta'] = 0.11  # 曾经设置过0.02
    params['max_depth'] = 5  # 曾经设置过4
    params["seed"] = random_seed

    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    train_qs = pd.Series(train['question1'].tolist() + train['question2'].tolist()).astype(str)
    words = (" ".join(train_qs)).lower().split()
    counts = Counter(words)
    weights = {word: get_weight(count) for word, count in counts.items()}
    stops = set(stopwords.words("english"))

    df = pd.concat([train, test])
    df['word_shares'] = df.apply(word_share, axis=1, raw=True)

    x = pd.DataFrame()
    x['word_match'] = df['word_shares'].apply(lambda x: float(x.split(':')[0]))
    x['word_match_2root'] = np.sqrt(x['word_match'])
    x['tfidf_word_match'] = df['word_shares'].apply(lambda x: float(x.split(':')[1]))
    x['shared_count'] = df['word_shares'].apply(lambda x: float(x.split(':')[2]))

    x['stops1_ratio'] = df['word_shares'].apply(lambda x: float(x.split(':')[3]))
    x['stops2_ratio'] = df['word_shares'].apply(lambda x: float(x.split(':')[4]))
    x['shared_2gram'] = df['word_shares'].apply(lambda x: float(x.split(':')[5]))
    x['cosine'] = df['word_shares'].apply(lambda x: float(x.split(':')[6]))
    x['words_hamming'] = df['word_shares'].apply(lambda x: float(x.split(':')[7]))
    x['diff_stops_r'] = x['stops1_ratio'] - x['stops2_ratio']

    x['len_q1'] = df['question1'].apply(lambda x: len(str(x)))
    x['len_q2'] = df['question2'].apply(lambda x: len(str(x)))
    x['diff_len'] = x['len_q1'] - x['len_q2']

    x['caps_count_q1'] = df['question1'].apply(lambda x: sum(1 for i in str(x) if i.isupper()))
    x['caps_count_q2'] = df['question2'].apply(lambda x: sum(1 for i in str(x) if i.isupper()))
    x['diff_caps'] = x['caps_count_q1'] - x['caps_count_q2']

    x['len_char_q1'] = df['question1'].apply(lambda x: len(str(x).replace(' ', '')))
    x['len_char_q2'] = df['question2'].apply(lambda x: len(str(x).replace(' ', '')))
    x['diff_len_char'] = x['len_char_q1'] - x['len_char_q2']

    x['len_word_q1'] = df['question1'].apply(lambda x: len(str(x).split()))
    x['len_word_q2'] = df['question2'].apply(lambda x: len(str(x).split()))
    x['diff_len_word'] = x['len_word_q1'] - x['len_word_q2']

    x['avg_world_len1'] = x['len_char_q1'] / x['len_word_q1']
    x['avg_world_len2'] = x['len_char_q2'] / x['len_word_q2']
    x['diff_avg_word'] = x['avg_world_len1'] - x['avg_world_len2']

    x['exactly_same'] = (df['question1'] == df['question2']).astype(int)
    x['duplicated'] = df.duplicated(['question1', 'question2']).astype(int)
    add_word_count(x, df, 'how')
    add_word_count(x, df, 'what')
    add_word_count(x, df, 'which')
    add_word_count(x, df, 'who')
    add_word_count(x, df, 'where')
    add_word_count(x, df, 'when')
    add_word_count(x, df, 'why')

    print(x.columns)
    print(x.describe())

    feature_names = list(x.columns.values)
    create_feature_map(feature_names)
    print("Features: {}".format(feature_names))

    x_train = x[:df_train.shape[0]]
    x_test = x[df_train.shape[0]:]
    y_train = df_train['is_duplicate'].values
    del x, df_train

    if 1:  # Now we oversample the negative class - on your own risk of overfitting!
        pos_train = x_train[y_train == 1]
        neg_train = x_train[y_train == 0]

        print("Oversampling started for proportion: {}".format(len(pos_train) / (len(pos_train) + len(neg_train))))
        p = 0.165
        scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
        while scale > 1:
            neg_train = pd.concat([neg_train, neg_train])
            scale -= 1
        neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
        print("Oversampling done, new proportion: {}".format(len(pos_train) / (len(pos_train) + len(neg_train))))

        x_train = pd.concat([pos_train, neg_train])
        y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()
        del pos_train, neg_train

    print("Training data: X_train: {}, Y_train: {}, X_test: {}".format(x_train.shape, len(y_train), x_test.shape))
    clr = train_xgb(x_train, y_train, params)
    preds = predict_xgb(clr, x_test)

    print("Writing output...")
    sub = pd.DataFrame()
    sub['test_id'] = df_test['test_id']
    sub['is_duplicate'] = preds * .75
    sub.to_csv("xgb_seed{}_n{}.csv".format(RS, ROUNDS), index=False)

    print("Features importances...")
    importance = clr.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=operator.itemgetter(1))
    ft = pd.DataFrame(importance, columns=['feature', 'fscore'])

    ft.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(10, 25))
    plt.gcf().savefig('features_importance.png')


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

bst = train_xgb(x_train, y_train, params)

sub = pd.DataFrame({'test_id': test['test_id'], 'is_duplicate': p})
sub.to_csv(result_file, index=False)
