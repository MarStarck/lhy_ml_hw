# coding:utf-8
__author__ = 'Mars'

import sys
import pandas as pd
import numpy as np

IN_FILE_PATH = 'data/train.csv'
NR = 0
FEATURE_LEN = 18

def csv_loader(in_path, N = 8):
    '''
    load csv data and process into x, y
    :param in_path:
    :param N:
    :return:
    '''
    # load data frame
    df = pd.read_csv(in_path,encoding='big5', skiprows=0, usecols=range(3, 27))
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

if __name__ == '__main__':
    if len(sys.argv) > 1:
        in_path = sys.argv[1]
    else:
        in_path = IN_FILE_PATH
    csv_loader(in_path)



