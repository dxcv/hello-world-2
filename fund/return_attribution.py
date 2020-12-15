# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 11:09:58 2020
基金的业绩归因，挖掘基金的选股能力
@author: Administrator
"""
from __future__ import division
import statsmodels.api as sm
import pandas as pd
import numpy as np
import datetime
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta

from jqdatasdk import *

# auth('18610039264', 'zg19491001')
auth('15168322665', 'Juzheng2018')
# 提取符合条件的基金名单
def fund_find(start_day, operate_mode, underlying_asset_type):
    q = query(finance.FUND_MAIN_INFO).filter(finance.FUND_MAIN_INFO.operate_mode_id == operate_mode,
                                             finance.FUND_MAIN_INFO.underlying_asset_type_id == underlying_asset_type,
                                             finance.FUND_MAIN_INFO.start_date < start_day)
    df = finance.run_query(q)
    print('一共' + str(len(df)) + '只基金')
    return (df)

# 提取基金净值数据
def fund_value(start_day, code):
    q = query(finance.FUND_NET_VALUE.code,
              finance.FUND_NET_VALUE.day,
              finance.FUND_NET_VALUE.sum_value,
              finance.FUND_NET_VALUE.refactor_net_value).filter(finance.FUND_NET_VALUE.code == code,
                                                                finance.FUND_NET_VALUE.day > start_day)
    df = finance.run_query(q)
    return (df)

# 获取价格
def stock_price(sec, period, sday, eday):
    """
    输入 股票代码，开始日期，截至日期
    输出 个股的后复权的开高低收价格
    """
    temp = get_price(sec, start_date=sday, end_date=eday, frequency=period, fields=['open', 'close', 'high', 'low'],
                     skip_paused=True, fq='pre', count=None).reset_index() \
        .rename(columns={'index': 'day'})
    temp['code'] = sec
    return temp

if __name__ == '__main__':

    print("V")
    # 确定基金池，选择上市时间大于3年的基金
    today = datetime.datetime.today()
    today = str(today)[:10]
    fund_stime = str(datetime.datetime.today() - relativedelta(months=60))[:10]  # 基金池开始时间

    operate_mode_id = [401001, 401006]
    underlying_asset_type_id = [402001, 402004]

    # fund_id 为符合条件的基金名单
    ret = list()
    for i in operate_mode_id:
        for j in underlying_asset_type_id:
            tmp = fund_find(fund_stime, i, j)
            ret.append(tmp)
    fund_id = pd.concat(ret)

    fund_id['end_date'] = fund_id['end_date'].fillna(1)
    fund_id = fund_id[fund_id['end_date'] == 1]
    
    # 获取基金近3年的净值
    # 采集基金净值数据，数据为
    fund_id_lst = fund_id.main_code.tolist()
    ret = list()
    for i in fund_id_lst:
        print(i)
        try:
            tmp = fund_value(fund_stime, i)
            ret.append(tmp)
        except:
            pass
    df = pd.concat(ret)
    df.to_csv('data/fund_value_' + today +'.csv',encoding='gbk')
    df = pd.read_csv('data/fund_value_' + today +'.csv', encoding='gbk',index_col=0)
    df.dropna(inplace=True)
    # 计算日收益率
    df.sort_values(by=['code','day'],inplace=True)
    df['month'] = df.day.apply(lambda s: str(s)[:7])
    
    ret_daily_rev = list()
    for idx, group in df.groupby('code'):
        tmp = group.copy()
        tmp['chg'] = tmp.sum_value.diff() / tmp.sum_value.shift(1)
        ret_daily_rev.append(tmp)
    df_rev_day = pd.concat(ret_daily_rev)
    df_rev_day.fillna(0,inplace=True)
    # 计算基金月收益率
    df_rev_month = df_rev_day.groupby(['code','month']).apply(lambda x : (1 + x.chg).prod() - 1)
    df_rev_month = df_rev_month.reset_index().rename(columns = {0:'month_chg'})
    # 计算业绩基准月收益率 
    benchmark = stock_price('000300.XSHG', '1d', fund_stime, today)
    benchmark['index_chg'] = benchmark.close.diff() / benchmark.close.shift(1)
    benchmark.fillna(0,inplace=True)
    benchmark['month'] = benchmark.day.apply(lambda s: str(s)[:7])
    benchmark = benchmark.groupby(['code','month']).apply(lambda x : (1 + x.index_chg).prod() - 1)
    benchmark = benchmark.reset_index().rename(columns = {0:'index_month_chg'})
    
    # 计算HM模型
    HM_window = 24
    rf = 0.03/12
    HM = list()
    for idx, group in df_rev_month.groupby('code'):
        try:
            print(idx)
            fund_HM = list()
            tmp = group.copy()
            tmp = tmp.merge(benchmark[['month','index_month_chg']])
            alpha_lst = list()
            p_lst = list()
            for i in range(HM_window,len(tmp)-1):
                tmp_x = tmp.iloc[i-HM_window:i]
                tmp_x['y'] = tmp_x['month_chg'] - rf
                tmp_x['x1'] = tmp_x['index_month_chg'] - rf
                tmp_x['x2'] = tmp_x['x1'].apply(lambda s: max(s, 0))
                x_tm = tmp_x[['x1', 'x2']]
                x_tm = sm.add_constant(x_tm)
                model_tm = sm.OLS(tmp_x['y'], x_tm).fit()
                [alpha_tm, beta1_tm, beta2_tm] = model_tm.params
                [p1_tm, p2_tm, p3_tm] = model_tm.pvalues
                alpha_lst.append(alpha_tm)
                p_lst.append(p1_tm)
            alpha_mean = np.mean(alpha_lst)
            p_ratio = len([i for i in p_lst if i < 0.1]) / len(p_lst)
            fund_HM.append(idx)
            fund_HM.append(alpha_mean)  
            fund_HM.append(p_ratio) 
            HM.append(fund_HM)
        except:
            print(str(idx) + ' is error!')
    out_HM = pd.DataFrame(HM,columns=['code','alpha_mean','p_ratio'])
    out_HM.to_csv("HM_model_60m.csv",encoding='gbk')
    # 收益的行业分解
    # 重仓股是否经常变更
    
    
    
    
