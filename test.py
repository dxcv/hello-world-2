import pandas as pd
import numpy as np
import time
from math import *
import numexpr as ne


def cal_1(x):
    return abs(cos(x)) ** 0.5 + sin(2 + 3 * x)


if __name__ == '__main__':
    print("V")
    start = time.time()
    a = range(1000000)
    ret = list()
    for i in a:
        ret.append(cal_1(i))
    end = time.time()
    print(("循环运行时间:%.2f秒" % (end - start)))
    #
    start = time.time()
    a = range(1000000)


    def f6(x):
        ex = 'abs(cos(x)) ** 0.5 + sin(2 + 3 * x)'
        ne.set_num_threads(10)
        return ne.evaluate(ex)


    result = f6(a)
    end = time.time()
    print(("循环运行时间:%.2f秒" % (end - start)))
