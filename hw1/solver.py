# coding:utf-8
__author__ = 'Mars'

import math
import numpy as np
from numpy.linalg import inv

def grad_des(x, y):
    # init params
    w = np.zeros(len(x[0]))
    l_rate = 10
    repeat = 10000
    # training
    x_t = x.transpose()
    s_gra = np.zeros(len(x[0]))

    for i in range(repeat):
        hypo = np.dot(x, w)
        loss = hypo - y
        cost = np.sum(loss ** 2) / len(x)
        cost_a = math.sqrt(cost)
        gra = np.dot(x_t, loss)
        s_gra += gra ** 2
        ada = np.sqrt(s_gra)
        w = w - l_rate * gra / ada
        # print ('iteration: %d | Cost: %f  ' % (i, cost_a))
    return w


def closed_form(x, y):
    w = np.matmul(np.matmul(inv(np.matmul(x.transpose(), x)), x.transpose()), y)
    return w