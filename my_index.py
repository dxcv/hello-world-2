# -*- coding: utf-8 -*-
"""
Created on Sat May 23 16:35:56 2020
计算4，5星基金各大类的大类指数
以及对比指定个股或者或者基金的收益曲线
@author: Administrator
"""

from __future__ import division
import pandas as pd
import datetime
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决符号无法显示
from jqdatasdk import *

auth('18610039264', 'zg19491001')


# 获取价格
def stock_price(sec, period, sday, eday):
    temp = get_price(sec, start_date=sday, end_date=eday, frequency=period, fields=['open', 'close', 'high', 'low'],
                     skip_paused=True, fq='pre', count=None).reset_index() \
        .rename(columns={'index': 'tradedate'})
    temp['stockcode'] = sec
    return temp


if __name__ == '__main__':
    # 定义常量
    period = 'daily'
    sday = '2015-01-01'
    eday = str(datetime.datetime.today())[:10]
    mode = 1
    if mode == 1:
        # 读取股票池
        data = pd.read_csv('lfp_good_stocks.csv', index_col=0, encoding='gbk')
        data = data[data['category'] != '其他']
        stock_lst = data.stockcode.tolist()
        stock_lst = data.stockcode.tolist()

        # 统计净值收益率
        ret = list()
        for i in stock_lst:
            print(i + " is get value")
            tmp = stock_price(i, period, sday, eday)
            tmp.sort_values(by='tradedate', inplace=True)
            tmp = tmp[['stockcode', 'tradedate', 'close']]
            tmp.close = tmp.close.diff() / tmp.close
            ret.append(tmp)
        df = pd.concat(ret)
        # 贴大类标签
        df_all_category = pd.merge(df, data[['stockcode', 'category']], on='stockcode')
        df_all_category.dropna(inplace=True)
        df_all_category.tradedate = df_all_category.tradedate.apply(lambda s: str(s)[:10])
        df_all_category = df_all_category.query("tradedate>'2019-01-01'")
        df_cal_index = df_all_category.groupby(['tradedate', 'category']).mean().reset_index()
        df_cal_index_pivot = pd.pivot_table(df_cal_index, index='tradedate', columns='category', values='close')
        df_cal_index_category = (1 + df_cal_index_pivot).cumprod()
        df_cal_index_category.plot()
    elif mode == 0:
        stock_lst = ['600585', '002043']
        stock_lst = [normalize_code(i) for i in stock_lst]
        # 统计净值收益率
        ret = list()
        for i in stock_lst:
            print(i + " is get value")
            tmp = stock_price(i, period, sday, eday)
            tmp.sort_values(by='tradedate', inplace=True)
            tmp = tmp[['stockcode', 'tradedate', 'close']]
            tmp.close = tmp.close.diff() / tmp.close
            ret.append(tmp)
        df = pd.concat(ret)
        df.dropna(inplace=True)
        df.tradedate = df.tradedate.apply(lambda s: str(s)[:10])
        df = df.query("tradedate>'2017-01-01'")
        df = pd.pivot_table(df, index='tradedate', columns='stockcode', values='close')
        df = (1 + df).cumprod()
        df.plot()
