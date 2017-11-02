# coding:utf-8
__author__ = 'Mars'

import sys
import pandas as pd
import numpy as np

IN_FILE_PATH = 'data/train.csv'
TEST_FILE_PATH = 'data/test.csv'
ANS_FILE_PATH = 'data/ans.csv'

NR = 0
FEATURE_LEN = 18

def csv_loader(in_path, N = 8, c_s = 3, c_e = 27, sr = 0, encoding='big5'):
    '''
    load csv data and process into x, y
    :param in_path:
    :param N:
    :return:
    '''
    # load data frame
    df = pd.read_csv(in_path,encoding=encoding, skiprows=sr, usecols=range(c_s, c_e))
    data = df.values
    # replace NR with 0
    data[data=='NR'] = NR
    data = data.astype(float)
    # split according to N day
    r,c = data.shape
    x = []
    y = []
    for i in range(0, c - N):
        for j in range(0, r/FEATURE_LEN):
            _feature = data[j: j+FEATURE_LEN, i:i+N]
            _x = _feature.flatten()
            _y = data[j+9, i+N]
            # print _feature, _y
            x.append(_x.tolist())
            y.append(float(_y))
    x = np.array(x)
    y = np.array(y)
    x_normed = x / x.max(axis=0)
    return x_normed, y

def test_loader(in_path=TEST_FILE_PATH, c_s = 2, c_e = 10, encoding='big5'):
    df = pd.read_csv(in_path, encoding=encoding, header=None, usecols=range(c_s, c_e))
    data = df.values
    # replace NR with 0
    data[data == 'NR'] = NR
    data = data.astype(float)
    # split according to N day
    r, N = data.shape
    assert r == 4320 and N == 8
    x = []
    for j in range(0, r / FEATURE_LEN):
        _feature = data[j: j + FEATURE_LEN, 0: N]
        _x = _feature.flatten()
        # print _feature, _y
        x.append(_x.tolist())
    x = np.array(x)
    x_normed = x / x.max(axis=0)
    return x_normed

if __name__ == '__main__':
    if len(sys.argv) > 1:
        in_path = sys.argv[1]
    else:
        in_path = IN_FILE_PATH
    x, y = csv_loader(in_path)
    x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
    print x



