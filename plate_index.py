# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 10:26:59 2020
计算生成板块指数
@author: Administrator
"""

from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import datetime
from dateutil.parser import parse
from jqdatasdk import *
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

# 提取成分股列表
def constituent_stock(df):
    df.stockcode = df.stockcode.apply(lambda s: normalize_code(s))
    return df.stockcode.drop_duplicates().tolist()

def get_prices(stock, s_t, e_t):
    return stock_price(stock,'daily',s_t,e_t)
    
def generate_stocks_price(stock_list, s_t):
    e_t = str(datetime.datetime.today())[:10]
    ret = get_prices(stock_list[0], s_t, e_t)
    ret['chg'] = ret.close.diff() / ret.close
    for i in stock_list[1:]:
        tmp = get_prices(i, s_t, e_t)
        tmp['chg'] = tmp.close.diff() / tmp.close
        ret = pd.concat([ret, tmp])
    return ret

def generate_index(stocks_price):
    tmp = stocks_price[['tradedate', 'chg']].groupby('tradedate')\
          .mean().reset_index()
    tmp.chg.fillna(0, inplace=True)
    tmp['index_'] = (1 + tmp['chg']).cumprod()
    return tmp

def auto_generate_index(stock_list, s_t):
    tmp = generate_stocks_price(stock_list, s_t)
    ret = generate_index(tmp)
    return ret

if __name__ == '__main__':
    new_energe = pd.read_csv('data/new_energe.csv')
    new_energe = constituent_stock(new_energe)
    
    chip = pd.read_csv('data/chip.csv')
    chip = constituent_stock(chip)
    
    internet = pd.read_csv('data/internet.csv')
    internet = constituent_stock(internet)
    
    tech = pd.read_csv('data/tech.csv')
    tech = constituent_stock(tech)
    
    ai = pd.read_csv('data/ai.csv')
    ai = constituent_stock(ai)
    
    
    start_day = '2015-01-01'
    # 生成制定行业指数
    new_energe_index = auto_generate_index(new_energe, start_day)
    
    
   
    