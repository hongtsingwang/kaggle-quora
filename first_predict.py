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

reload(sys)
sys.setdefaultencoding('utf-8')

# parser = argparse.ArgumentParser()
# parser.add_argument()
# args = parser.parse_args()

# output = args.output
logging.basicConfig(
    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
    level=logging.DEBUG,
    datefmt='%a, %d %b %Y %H:%M:%S'
)

home_dir = os.getcwd()
data_dir = os.path.join(home_dir, "data")

train_file = os.path.join(data_dir, "train.csv")
test_file = os.path.join(data_dir, "test.csv")

df_train = pd.read_csv(train_file)
df_test = pd.read_csv(test_file)

train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)
test_qs = pd.Series(df_test['question1'].tolist() + df_test['question2'].tolist()).astype(str)

dist_train = train_qs.apply(lambda x: len(x.split(' ')))
dist_test = test_qs.apply(lambda x: len(x.split(' ')))

from nltk.corpus import stopwords

stops = set(stopwords.words("english"))

print stops


def word_match_share(row):
    """
    计算相同的词的个数
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
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2)) / (len(q1words) + len(q2words))
    return R


def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row["question1"]).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row["question2"]).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    return (0.5 * len(shared_words_in_q1) / len(q1words) + 0.5 * len(shared_words_in_q2) / len(q2words))


test = pd.read_csv(test_file)
stops = set(stopwords.words("english"))

sub = pd.DataFrame()
sub['test_id'] = test['test_id']
sub["is_duplicate"] = test.apply(word_match_share, axis=1, raw=True)
sub.to_csv("count_words_benchmark.csv", index=False)

# # 数据统计和分析部分
# df_train_number = len(df_train)
# logging.info("item numbers of train file is %d" % df_train_number)
# df_train_duplicate_num = len(df_train[df_train["is_duplicate"] == 1])
# df_train_duplicate_ratio = float(df_train_duplicate_num * 1.0 / df_train_number)
# logging.info('Duplicate pairs number: %d, the ratio is %.3f%%' % (df_train_duplicate_num, df_train_duplicate_ratio*100))
#
# qids = pd.Series(df_train['qid1'].tolist() + df_train['qid2'].tolist())
# logging.info('Total number of questions in the training data: %d  ' % (len(np.unique(qids))))


from sklearn.metrics import log_loss

p = df_train['is_duplicate'].mean()  # Our predicted probability
# print('Predicted score:', log_loss(df_train['is_duplicate'], np.zeros_like(df_train['is_duplicate']) + p))
# print('Total number of question pairs for testing: {}'.format(len(df_test)))

import xgboost as xgb

# Set our parameters for xgboost
params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.02
params['max_depth'] = 4

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)

sub = pd.DataFrame({'test_id': df_test['test_id'], 'is_duplicate': p})
sub.to_csv('naive_submission.csv', index=False)
# import zipfile

# result = zipfile.ZipFile("naive_submission.zip", mode="w", compression=zipfile.ZIP_STORED)
# result.write('naive_submission.csv')
# result.close()
