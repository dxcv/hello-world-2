#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 15:47:54 2019
根据奇异谱分析
@author: yeecall
"""

from __future__ import division
import pandas as pd
import os
#import talib as tb
import matplotlib.pyplot as plt
import math
import numpy as np
from jqdatasdk import *
import datetime
auth('18610039264','zg19491001')


# 获取价格
def stock_price(sec,period,sday,eday):
  """
  输入 股票代码，开始日期，截至日期
  输出 个股的后复权的开高低收价格
  """
  temp= get_price(sec, start_date=sday, end_date=eday, frequency=period, fields=['open', 'close', 'high', 'low'], skip_paused=False, fq='pre', count=None).reset_index()\
                     .rename(columns={'index':'tradedate'})
  temp['stockcode']=sec
  return temp



# 嵌入
def getWindowMatrix(inputArray,t,m):
    temp = []
    n = t - m + 1
    for i in range(n):
        temp.append(inputArray[i:i+m])
    WindowMatrix = np.array(temp)
    return WindowMatrix

# 奇异谱分析，取第一主成分分量，返回重构矩阵
def SVDreduce(WindowMatrix):
    u,s,v = np.linalg.svd(WindowMatrix) #svd分解
    m1,n1 = u.shape
    m2,n2 = v.shape
    index = s.argmax()
    u1 = u[:,index]
    v1 = v[index]
    u1 = u1.reshape((m1,1))
    v1 = v1.reshape((1,n2))
    value = s.max()
    newMatrix = value*(np.dot(u1,v1))  #重构矩阵
    return newMatrix

# 对角线平均法重构序列
def recreateArray(newMatrix,t,m):
    ret = []
    n = t - m +1
    for p in range(1,t+1):
        if p<m:
            alpha = p
        elif p>t-m+1:
            alpha = t-p+1
        else:
            alpha = m
        sigma = 0
        for j in range(1,m+1):
            i = p - j +1
            if i>0 and i<n+1:
                sigma += newMatrix[i-1][j-1]
        ret.append(sigma/alpha)
    return ret

# 按不同的序列、不同的窗口大小计算SSA
def SSA(inputArray,t,m):
    WindowMatrix = getWindowMatrix(inputArray,t,m)
    newMatrix    = SVDreduce(WindowMatrix)
    newArray     = recreateArray(newMatrix,t,m)
    return newArray

if __name__=='__main__':
    
#    py_path=r'/Users/yeecall/Documents/mywork/指标/波段指标'
#    os.chdir(py_path)
    
    start_date='2018-01-01'
    end=datetime.datetime.today()
    end_date=str(end)[:10]
    M=20

    flag=1
    if flag==0:
          #数字货币
          HB10=get_exsymbol_kline('HUOBIPRO', 'huobi10', "1day", start_date, end_date)[2]
          BTC=get_exsymbol_kline('HUOBIPRO', 'btcusdt', "1day", start_date, end_date)[2]
          HB10=HB10[['date','open','high','low','close']]
          BTC=BTC[['date','open','high','low','close']]
          df=HB10.loc[:,['date','close']].merge(BTC.loc[:,['date','close']],on=['date'])
          df.columns=['date','HB10','BTC']
          df['BTC_ssa']=SSA(df.BTC.values,len(df),M)
          df.tail(500).loc[:,['BTC','BTC_ssa']].plot()
    else:
          #股票行情
          kind=['000016.XSHG','000300.XSHG','000905.XSHG','399678.XSHE']
          period='1d'
          stock=stock_price(kind[1],period,start_date,end_date)
          stock['ssa']=SSA(stock.close.values,len(stock),M)
          stock.tail(200000).loc[:,['close','ssa']].plot()
