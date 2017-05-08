# -*- coding:utf-8 -*-
# =======================================================
# 
# @FileName  : main.py.py
# @Author    : Wang Hongqing
# @Date      : 2017-05-05 19:33
# 
# =======================================================

import os
import sys
import argparse
import logging
from commands import getstatusoutput
from sklearn.feature_extraction import text
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer
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
data_dir = os.path.join(home_dir, "Data")
result_dir = os.path.join(home_dir, "Result")

train_file = os.path.join(data_dir, "train.csv")
test_file = os.path.join(data_dir, "test.csv")
train_df = pd.read_csv(train_file, header=0)
test_df = pd.read_csv(train_file, header=0)

porter = PorterStemmer()
snowball = SnowballStemmer('english')


def stem_str(x, stemmer=SnowballStemmer('english')):
    x = text.re.sub("[^a-zA-Z0-9]", " ", x)
    x = (" ").join([stemmer.stem(z) for z in x.split(" ")])
    x = " ".join(x.split())
    return x


def generate_stem():
    train_df["question1_porter"] = train_df["question1"].astype("str").apply(lambda x: stem_str(x.lower(), porter))
    train_df["question2_porter"] = train_df["question2"].astype("str").apply(lambda x: stem_str(x.lower(), porter))
    test_df["question1_porter"] = test_df["question1"].astype("str").apply(lambda x: stem_str(x.lower(), porter))
    test_df["question2_porter"] = test_df["question2"].astype("str").apply(lambda x: stem_str(x.lower(), porter))

    train_result_path = os.path.join(data_dir, "transform_data", "train_porter.csv")
    test_result_path = os.path.join(data_dir, "transform_data", "test_porter.csv")
    train_df.to_csv(train_result_path)
    test_df.to_csv(test_result_path)
    test_df.to_csv(test_result_path)


def test_function():
    logging.info("start testing generate_stem")


if __name__ == "__main__":
    test_function()
