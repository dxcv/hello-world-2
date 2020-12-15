# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 17:27:46 2020
观察沪深300+中证500+上证50股票池中，动量 VS 反转表现
@author: Administrator
"""

from __future__ import division
import pandas as pd
import numpy as np
import datetime
import talib as tb
from jqdatasdk import *

auth('18610039264', 'zg19491001')

# 获取当前各大指数的成分股
def index_stocks(_index):
  """
  输入 指数编码：000016.XSHG	上证50；000300.XSHG	沪深300；399005.XSHE	中小板指
               399006.XSHE	创业板指；000905.XSHG	中证500
  返回 成分股代码列表
  输出格式 list
  """
  return get_index_stocks(_index)

# 获取价格
def stock_price(sec, period, sday, eday):
    temp = get_price(sec, start_date=sday, end_date=eday, frequency=period, fields=['open', 'close', 'high', 'low'],
                     skip_paused=True, fq='pre', count=None).reset_index() \
        .rename(columns={'index': 'tradedate'})
    temp['stockcode'] = sec
    return temp


if __name__ == '__main__':
    SZ50_stocks_list=index_stocks('000016.XSHG')
    HS300_stocks_list=index_stocks('000300.XSHG')
    zz500_stocks_list=index_stocks('000905.XSHG')
    all_stocks = SZ50_stocks_list + HS300_stocks_list + zz500_stocks_list
    all_stocks =list(set(all_stocks))
    
    # 开始时间
    s_day = '2019-06-01'
    e_day = str(datetime.datetime.today())[:10]
    price_df =list()
    for s in all_stocks:
        print(str(s) + ' is cal price')
        price_df.append(stock_price(s,'1d',s_day,e_day))
    price = pd.concat(price_df)
    price = price.assign(low_n = lambda df:tb.LLV(df.close.values,30)) 
        
    
    
    
    
    

