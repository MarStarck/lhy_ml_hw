# coding:utf-8
__author__ = 'Mars'

import sys
import pandas as pd

IN_FILE_PATH = 'data/train.csv'
NR = 0

def csv_loader(in_path):
    # load data frame
    df = pd.read_csv(in_path,encoding='big5', skiprows=0, usecols=range(3, 27))
    data = df.values
    print data[10][0]
    # replace NR with 0
    data[data=='NR'] = NR
    # TODO: split

if __name__ == '__main__':
    if len(sys.argv) > 1:
        in_path = sys.argv[1]
    else:
        in_path = IN_FILE_PATH
    csv_loader(in_path)



