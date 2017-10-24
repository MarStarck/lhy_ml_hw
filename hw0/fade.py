# coding:utf-8
__author__ = 'Mars'

import sys
from PIL import Image
import numpy as np

IN_FILE_PATH = 'data/westbrook.jpg'
OUT_FILE_PATH = 'Q2.jpg'


def fade_img(in_path):
    # load img
    img = Image.open(in_path)
    img_array = np.array(img)
    res_img = Image.fromarray(img_array/2)
    res_img.save(OUT_FILE_PATH)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        in_path = sys.argv[1]
    else:
        in_path = IN_FILE_PATH
        fade_img(in_path)



