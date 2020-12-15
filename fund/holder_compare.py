# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:14:20 2020
计算基金期M日持仓模拟净值与实际净值对比
@author: Administrator
"""

from __future__ import division
import pandas as pd
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决符号无法显示
from jqdatasdk import *
auth('18610039264','zg19491001')


# 提取基金净值数据
def fund_value(start_day,code):
      q=query(finance.FUND_NET_VALUE.code,
              finance.FUND_NET_VALUE.day,
              finance.FUND_NET_VALUE.sum_value,
              finance.FUND_NET_VALUE.refactor_net_value).filter(finance.FUND_NET_VALUE.code==code,
              finance.FUND_NET_VALUE.day>= start_day)
      df=finance.run_query(q)
      return(df)
  
def fund_hold_(code, date):
    q = query(finance.FUND_PORTFOLIO_STOCK.code, 
              finance.FUND_PORTFOLIO_STOCK.period_end,
              finance.FUND_PORTFOLIO_STOCK.symbol,
              finance.FUND_PORTFOLIO_STOCK.name,
              finance.FUND_PORTFOLIO_STOCK.proportion).filter(finance.FUND_PORTFOLIO_STOCK.code==code,
                                                              finance.FUND_PORTFOLIO_STOCK.period_end==date)\
                                                              .order_by(finance.FUND_PORTFOLIO_STOCK.proportion.desc()).limit(20)
    df = finance.run_query(q)
    return df

# 获取持股比例
def stock_ratio_(code, date):
    q = query(finance.FUND_PORTFOLIO.code,
              finance.FUND_PORTFOLIO.period_end,
              finance.FUND_PORTFOLIO.stock_rate).filter(finance.FUND_PORTFOLIO.code==code,finance.FUND_PORTFOLIO.period_end == date)
    ss = finance.run_query(q)
    return ss

# 获取价格
def stock_price(sec,period,sday,eday):
  """
  输入 股票代码，开始日期，截至日期
  输出 个股的后复权的开高低收价格
  """
  temp= get_price(sec, start_date=sday, end_date=eday, frequency=period, fields=['open', 'close', 'high', 'low'], skip_paused=True, fq='pre', count=None).reset_index()\
                     .rename(columns={'index':'tradedate'})
  temp['stockcode']=sec
  return temp

if __name__ == '__main__':
    # 变量
    fund = '163412'
    pub_day = '2020-06-31'
    today = str(datetime.today())[:10]
    # 基金净值
    fund_net = fund_value(pub_day,fund)
    fund_net['fund_chg'] = fund_net.sum_value.diff() / fund_net.sum_value
    # 基金的持股比例
    sr = stock_ratio_(fund, pub_day)
    stock_ratio = sr['stock_rate'].values[0] / 100
    # 获取持股信息
    holds = fund_hold_(fund, pub_day)
    holds.drop_duplicates(subset=['symbol'], inplace=True)
    holds['q'] = holds['proportion'] / holds['proportion'].sum()
    # 计算收益率及净值对比
    stock_lst = holds.symbol.tolist()
    q_lst = holds.q.tolist()
    ret = list()
    for i in range(len(stock_lst)):
        print(i)
        f = stock_lst[i]
        q = q_lst[i]
        stk = normalize_code(f)
        stk_price = stock_price(stk, '1d', pub_day, today)
        stk_price['stk_chg'] = stk_price.close.diff() / stk_price.close
        stk_price['stk_chg_q'] = stk_price['stk_chg'] * q
        stk_price.dropna(inplace=True)
        ret.append(stk_price)
    df = pd.concat(ret)
    df_pivot = pd.pivot_table(df, index='tradedate',columns='stockcode', values='stk_chg_q')
    fund_moni = df_pivot.sum(axis=1).reset_index()
    fund_moni.columns = ['tradedate', 'moni_chg']
    fund_moni['moni_chg'] = fund_moni['moni_chg'] * stock_ratio
    fund_moni.tradedate = fund_moni.tradedate.apply(lambda s: str(s)[:10])
    
    fund_chg = fund_net[['day', 'fund_chg']].dropna()
    fund_chg.columns = ['tradedate', 'fund_chg']
    fund_chg.tradedate = fund_chg.tradedate.apply(lambda s: str(s)[:10])
    
    
    
    fund_compare = fund_chg.merge(fund_moni)
    fund_compare['trace_error'] = fund_compare['moni_chg'] - fund_compare['fund_chg']
    fund_compare['trace_error_rolling'] = fund_compare['trace_error'].rolling(5).std(ddof=1)
    fund_compare_value = fund_compare[['tradedate','fund_chg','moni_chg']]
    fund_compare = fund_compare.set_index('tradedate')
    fund_compare.plot(title=fund)
    

    fund_compare_value = fund_compare_value.set_index('tradedate')
    fund_compare_value = (1 + fund_compare_value).cumprod()  
    fund_compare_value.plot(title=fund + '_' + '净值比较')
