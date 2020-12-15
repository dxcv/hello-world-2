# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 11:45:14 2020
计算股价走势与股东人数的双轴图
@author: Administrator
"""

from __future__ import division
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决符号无法显示
from matplotlib.ticker import FuncFormatter
from jqdatasdk import *
auth('18610039264', 'zg19491001')


# 获取价格
def stock_price(sec, period, sday, eday):
    temp = get_price(sec, start_date=sday, end_date=eday, frequency=period, fields=['open', 'close', 'high', 'low'],
                     skip_paused=True, fq='pre', count=None).reset_index() \
        .rename(columns={'index': 'tradedate'})
    temp['stockcode'] = sec
    return temp

# 获取股东数据
def holder_num(stk,s_day):
    tmp = finance.run_query(query(finance.STK_HOLDER_NUM.code,
                                  finance.STK_HOLDER_NUM.end_date,
                                  finance.STK_HOLDER_NUM.share_holders)\
                      .filter(finance.STK_HOLDER_NUM.code==stk,
                    finance.STK_HOLDER_NUM.end_date >= s_day))
    return tmp
# 获取流通股本数据
def circul_cap(stk,e_day,num):
    q = query(valuation.code,
              valuation.day,
              valuation.circulating_cap,
            ).filter(valuation.code == stk)
    tmp = get_fundamentals_continuously(q, end_date=e_day, count=num)
    return tmp
    

if __name__ == "__main__":
    st_lst = ['603187','002050','000333','000002','002027','002410','300662','002271','300760','300012','300285','600763','300413','002352','600519','300783','300138']
    st = [normalize_code(i) for i in st_lst]
    
    
    
    
    stock_code = '300138' # 股票代码
    relative_delta_period = 5 # 滞后期
    num = 252 * relative_delta_period
    today = datetime.today()
    s_day = str(today - relativedelta(years=relative_delta_period))[:10]
    e_day = str(today)[:10]
    stock_code = normalize_code(stock_code)
    stock_name = get_security_info(stock_code).display_name
    #状态开关
    # 0 : 机会，原数据，越小说明筹码更加集中；
    # 1 ：风险，流通股本处理过的数据，越大说明筹码更加分散；
    flag = 0
    if flag == 1:
        # 获得股价
        prices = stock_price(stock_code, '1d', s_day, e_day)
        
        # 获得股东数据
        holders = holder_num(stock_code,s_day)
        
        # 获得流通股本数据
        cc = circul_cap(stock_code,e_day,num).minor_xs(stock_code)
        
        # 数据处理
        # 处理收盘价数据，保留每月最后一个交易日数据
        prices['M_O_D'] = prices.tradedate.apply(lambda s : str(s)[:7])
        ret_tail_d = list()
        for idx, group in prices.groupby("M_O_D"):
            tmp = group.sort_values(by='tradedate').copy()
            ret_tail_d.append(tmp.tail(1))
        price_tail = pd.concat(ret_tail_d)
        # 处理股东数据，获得年月数据
        holders['M_O_D'] = holders.end_date.apply(lambda s : str(s)[:7])
        # 处理流通股本数据
        cc['M_O_D'] = cc['day.1'].apply(lambda s : str(s)[:7])
        ret_tail_cc = list()
        for idx, group in cc.groupby("M_O_D"):
            tmp = group.sort_values(by='day.1').copy()
            ret_tail_cc.append(tmp.tail(1))
        cc_tail = pd.concat(ret_tail_cc)
        # 合数据
        
        df = price_tail[['M_O_D','close']].merge(holders[['M_O_D','share_holders']])
        df = df.merge(cc_tail[['M_O_D', 'circulating_cap']])
        df.iloc[:,-1] = df.iloc[:,-1] / df.iloc[0,-1]
        df['share_holders_adjust'] = df['share_holders'] / df['circulating_cap']
        df['share_holders_adjust'] = df['share_holders_adjust'] / 1000
        df['M_O_D'] = df['M_O_D'].apply(lambda s:s[2:4]+s[5:7])
        # df['M_O_D'] = df['M_O_D'].apply(lambda s:datetime.strptime(s,'%y%m'))
        # df['M_O_D'] = df['M_O_D'].apply(lambda s:s.strftime('%Y-%m','%y%m'))   
        # 画双轴图
        fig, ax1 = plt.subplots(figsize = (10, 5), facecolor='white')
        ax1.bar(df.M_O_D, df.share_holders_adjust, color='g', alpha=0.5)
        ax1.set_xlabel("月份")
        ax1.set_ylabel("股东户数（千户）")
        
        ax2 = ax1.twinx()
        ax2.plot(df.M_O_D, df.close, '-or')
        ax2.set_ylabel('股价')
        ax2.set_ylim(0,int((df.close.max()*1.5)))
        
        plt.title('%s的股东户数与股价关系图%s'%(stock_name,flag))
        plt.show()
    else:
        # 获得股价
        prices = stock_price(stock_code, '1d', s_day, e_day)
        
        # 获得股东数据
        holders = holder_num(stock_code,s_day)
        
        # 获得流通股本数据
        cc = circul_cap(stock_code,e_day,num).minor_xs(stock_code)
        
        # 数据处理
        # 处理收盘价数据，保留每月最后一个交易日数据
        prices['M_O_D'] = prices.tradedate.apply(lambda s : str(s)[:7])
        ret_tail_d = list()
        for idx, group in prices.groupby("M_O_D"):
            tmp = group.sort_values(by='tradedate').copy()
            ret_tail_d.append(tmp.tail(1))
        price_tail = pd.concat(ret_tail_d)
        # 处理股东数据，获得年月数据
        holders['M_O_D'] = holders.end_date.apply(lambda s : str(s)[:7])
        # 处理流通股本数据
        cc['M_O_D'] = cc['day.1'].apply(lambda s : str(s)[:7])
        ret_tail_cc = list()
        for idx, group in cc.groupby("M_O_D"):
            tmp = group.sort_values(by='day.1').copy()
            ret_tail_cc.append(tmp.tail(1))
        cc_tail = pd.concat(ret_tail_cc)
        # 合数据
        
        df = price_tail[['M_O_D','close']].merge(holders[['M_O_D','share_holders']])
        df = df.merge(cc_tail[['M_O_D', 'circulating_cap']])
        df.iloc[:,-1] = df.iloc[:,-1] / df.iloc[0,-1]
        df['share_holders'] = df['share_holders'] / df['circulating_cap']
        df['share_holders'] = df['share_holders'] / 1000
        df['M_O_D'] = df['M_O_D'].apply(lambda s:s[2:4]+s[5:7])
        # df['M_O_D'] = df['M_O_D'].apply(lambda s:datetime.strptime(s,'%y%m'))
        # df['M_O_D'] = df['M_O_D'].apply(lambda s:s.strftime('%Y-%m','%y%m'))   
        # 画双轴图
        fig, ax1 = plt.subplots(figsize = (10, 5), facecolor='white')
        ax1.bar(df.M_O_D, df.share_holders, color='g', alpha=0.5)
        ax1.set_xlabel("月份")
        ax1.set_ylabel("股东户数（千户）")
        
        ax2 = ax1.twinx()
        ax2.plot(df.M_O_D, df.close, '-or')
        ax2.set_ylabel('股价')
        ax2.set_ylim(0,int((df.close.max()*1.5)))
        
        plt.title('%s的股东户数与股价关系图%s'%(stock_name,flag))
        plt.show()
    
    
    
    