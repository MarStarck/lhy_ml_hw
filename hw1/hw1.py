# coding:utf-8
__author__ = 'Mars'

from data_loader import *
from solver import *
import csv

x, y = csv_loader(IN_FILE_PATH)
x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)

x_test = test_loader(TEST_FILE_PATH, c_s=2, c_e=10, encoding='utf-8')
x_test = np.concatenate((np.ones((x_test.shape[0], 1)), x_test), axis=1)
# print 'test shape', x_test.shape

def save_model(w):
    # save model
    np.save('model.npy',w)


def load_model():
    w = np.load('model.npy')
    return w


def training_with_grad():
    return grad_des(x, y)


def training_with_closed():
    return closed_form(x, y)

if __name__ == '__main__':
    w1 = training_with_closed()

    # compute weight
    w = training_with_grad()
    save_model(w)

    # calculate predict res
    ans = np.dot(x_test, w)
    print len(ans)
    filename = "data/predict.csv"
    text = open(filename, "w+")
    s = csv.writer(text, delimiter=',', lineterminator='\n')
    s.writerow(["id", "value"])
    for i in range(0,len(ans)):
        s.writerow(["id_%d"%i, ans[i]])
    text.close()

