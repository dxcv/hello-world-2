# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 18:44:02 2020
测试升哥BBI交易信号逻辑
@author: Administrator
"""

from __future__ import division
import pandas as pd
import numpy as np
import talib as tb
# from jqdatasdk import *
# auth('18610039264','zg19491001')

def bbi(df):
    tmp = df.copy()
    tmp = tmp.assign(MA3 = tb.MA(tmp.close.values, 3))\
             .assign(MA6 = tb.MA(tmp.close.values, 6))\
             .assign(MA12 = tb.MA(tmp.close.values, 12))\
             .assign(MA24 = tb.MA(tmp.close.values, 24))\
             .assign(BBI = lambda df: 0.25 * (df['MA3'] + df['MA6'] + df['MA12'] + df['MA24']))
    return tmp[['stockcode' , 'tradedate' , 'close', 'BBI']]
    
def bs(row):
    if (row['bbi_cross_sign_3'] == 2) & (row['bbi_sign_3'] == 3):
        return 'B'
    elif (row['bbi_cross_sign_3'] == -2) & (row['bbi_sign_3'] == -3):
        return 'S'
    else:
        return ' '

def trade_detail(code, df):
    price = df[df['stockcode'] == code].copy()
    tmp = price[['stockcode', 'tradedate', 'close']]
    
    tmp = bbi(tmp)
               
    tmp.dropna(inplace=True) 
    tmp = tmp.assign(bbi_bias = lambda df: df['close'] - df['BBI'])\
             .assign(bbi_sign = lambda df: df.bbi_bias.apply(lambda s: 1 if s>0 else -1))\
             .assign(bbi_sign_3 = lambda df: df.bbi_sign.rolling(3).sum())\
             .assign(bbi_sign_ref_1 = lambda df: df.bbi_sign.shift(1))\
             .assign(bbi_cross_sign = lambda df: df['bbi_sign'] - df['bbi_sign_ref_1'])\
             .assign(bbi_cross_sign_3 = lambda df: df.bbi_cross_sign.rolling(3).sum())

    bs_list =list()
    for idx, row in tmp.iterrows():
        if (row['bbi_cross_sign_3'] == 2) & (row['bbi_sign_3'] == 3):
            bs_list.append('B')
        elif (row['bbi_cross_sign_3'] == -2) & (row['bbi_sign_3'] == -3):
            bs_list.append('S')
        else:
            bs_list.append('') 
    tmp['BS'] = bs_list
    
    tmp = tmp[['stockcode' , 'tradedate' , 'close', 'BBI', 'BS']]
    ret= list()
    zt = ''
    tradeid = ''
    for idx, row_ in tmp.iterrows():
        res = list()
        if (row_['BS'] == 'B') & (zt != 'B'):
            zt = 'B'
            tradeid = idx
            res.append(tradeid)
            res.append(row_['tradedate'])
            res.append('B')
            res.append(row_['close'])
            ret.append(res)
        elif (row_['BS'] == 'S') & (zt == 'B'):
            zt = ''
            res.append(tradeid)
            res.append(row_['tradedate'])
            res.append('S')
            res.append(row_['close'])
            ret.append(res)
            
    trades = pd.DataFrame(ret, columns=['id', 'date', 'dire', 'price'])
            
    B = trades[trades['dire'] == 'B']
    S = trades[trades['dire'] == 'S']
    B = B[['id', 'date', 'price']].rename(columns = {'date':'buy_date','price':'buy_price'})
    S = S[['id', 'date', 'price']].rename(columns = {'date':'sell_date','price':'sell_price'})
    trade_detail = B.merge(S)
    trade_detail['code'] = code
    trade_detail['profit'] = (trade_detail['sell_price'] - trade_detail['buy_price']) / trade_detail['buy_price']
    return trade_detail

if __name__ == '__main__':
    # 导入行情数据
    df = pd.read_csv('long_niu_v2_stock_data.csv', index_col=0)
    # 代码列表
    code_lst = df.stockcode.drop_duplicates().tolist()
    # 以一个代码为例
    # code = code_lst[0] 
    # # 进入分析模块
    # price = df[df['stockcode'] == code]
    # tmp = price[['stockcode', 'tradedate', 'close']]
    
    # tmp = bbi(tmp)
               
    # tmp.dropna(inplace=True) 
    # tmp = tmp.assign(bbi_bias = lambda df: df['close'] - df['BBI'])\
    #          .assign(bbi_sign = lambda df: df.bbi_bias.apply(lambda s: 1 if s>0 else -1))\
    #          .assign(bbi_sign_3 = lambda df: df.bbi_sign.rolling(3).sum())\
    #          .assign(bbi_sign_ref_1 = lambda df: df.bbi_sign.shift(1))\
    #          .assign(bbi_cross_sign = lambda df: df['bbi_sign'] - df['bbi_sign_ref_1'])\
    #          .assign(bbi_cross_sign_3 = lambda df: df.bbi_cross_sign.rolling(3).sum())

    # bs_list =list()
    # for idx, row in tmp.iterrows():
    #     if (row['bbi_cross_sign_3'] == 2) & (row['bbi_sign_3'] == 3):
    #         bs_list.append('B')
    #     elif (row['bbi_cross_sign_3'] == -2) & (row['bbi_sign_3'] == -3):
    #         bs_list.append('S')
    #     else:
    #         bs_list.append('') 
    # tmp['BS'] = bs_list
    
    # tmp = tmp[['stockcode' , 'tradedate' , 'close', 'BBI', 'BS']]
    # ret= list()
    # zt = ''
    # tradeid = ''
    # for idx, row_ in tmp.iterrows():
    #     res = list()
    #     if (row_['BS'] == 'B') & (zt != 'B'):
    #         zt = 'B'
    #         tradeid = idx
    #         res.append(tradeid)
    #         res.append(row_['tradedate'])
    #         res.append('B')
    #         res.append(row_['close'])
    #         ret.append(res)
    #     elif (row_['BS'] == 'S') & (zt == 'B'):
    #         zt = ''
    #         res.append(tradeid)
    #         res.append(row_['tradedate'])
    #         res.append('S')
    #         res.append(row_['close'])
    #         ret.append(res)
            
    # trades = pd.DataFrame(ret, columns=['id', 'date', 'dire', 'price'])
            
    # B = trades[trades['dire'] == 'B']
    # S = trades[trades['dire'] == 'S']
    # B = B[['id', ,'date', 'price']].rename(columns = {'date':'buy_date','price':'buy_price'})
    # S = S[['id', ,'date', 'price']].rename(columns = {'date':'sell_date','price':'sell_price'})
    # trade_detail = B.merge(S)
    # trade_detail['code'] = code
    # trade_detail['profit'] = (trade_detail['sell_price'] - trade_detail['buy_price']) / trade_detail['buy_price']
    # trade_detail.to_excel('signal_analysis.xls')
    # tmp.to_excel('bbi_bs.xls')
    
    # ss = trade_detail(code, df)
    out = list()
    for i in code_lst:
        print(i + ' is cal')
        try:
            tmp = trade_detail(i, df)
            tmp.to_csv('yss/'+i+'.csv',encoding='gbk')
            out.append(tmp)
        except:
            print(i + ' is error')
    result = pd.concat(out)
    result.to_csv('bbi_all_stocks_stat.csv',encoding='gbk')
    
        
            
                 
        
        