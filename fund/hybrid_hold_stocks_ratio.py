# -*- coding: utf-8 -*-
"""
Created on Mon May 18 16:56:55 2020
将股票型、混合型基金、债券型基金进行分类
分为偏股混合 偏债混合以及 标准混合型
将基金的行业属性输出，便于寻找偏主题的基金
@author: Administrator
"""

from __future__ import division
import pandas as pd
import numpy as np
import statsmodels.api as sm
import math
import datetime
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta

# from statsmodels import regression
# import statsmodels.api as sm
from jqdatasdk import *

auth('18610039264', 'zg19491001')


# count_=get_query_count()
# print(count_)

# 提取符合条件的基金名单
def fund_find(start_day, operate_mode, underlying_asset_type):
    q = query(finance.FUND_MAIN_INFO).filter(finance.FUND_MAIN_INFO.operate_mode_id == operate_mode,
                                             finance.FUND_MAIN_INFO.underlying_asset_type_id == underlying_asset_type,
                                             finance.FUND_MAIN_INFO.start_date < start_day)
    df = finance.run_query(q)
    print('一共' + str(len(df)) + '只基金')
    return (df)

def check_industry(code, indust_classify_code='sw_l1'):
    try:
        return get_industry(code)[code][indust_classify_code]['industry_code']
    except:
        return 0
    
def cal_hybrid_hold_ratio():
    today = datetime.datetime.today()
    today = str(today)[:10]
    period = 36
    fund_stime = str(datetime.datetime.today() - relativedelta(months=period))[:10]  # 基金池开始时间

    operate_mode_id = [401001, 401006]
    underlying_asset_type_id = [402004]
    # fund_id 为符合条件的基金名单
    ret = list()
    for i in operate_mode_id:
        for j in underlying_asset_type_id:
            tmp = fund_find(fund_stime, i, j)
            ret.append(tmp)
    fund_id = pd.concat(ret)
    fund_id['end_date'] = fund_id['end_date'].fillna(1)
    fund_id = fund_id[fund_id['end_date'] == 1]
    
    fund_lst = fund_id.main_code.tolist()
    ret = list()
    for fd in fund_lst:
        print(fd + ' is cal')
        q = query(finance.FUND_PORTFOLIO.code,
                  finance.FUND_PORTFOLIO.period_end,
                  finance.FUND_PORTFOLIO.stock_rate).filter(finance.FUND_PORTFOLIO.code==fd).order_by(finance.FUND_PORTFOLIO.period_end.desc()).limit(12)
        ss = finance.run_query(q)
        ret.append(ss)
    fund_hold_ratio = pd.concat(ret)
    out = fund_hold_ratio.groupby(['code'])['stock_rate'].mean().reset_index()
    return out

def cal_bond_hold_ratio():
    today = datetime.datetime.today()
    today = str(today)[:10]
    period = 36
    fund_stime = str(datetime.datetime.today() - relativedelta(months=period))[:10]  # 基金池开始时间

    operate_mode_id = [401001, 401006]
    underlying_asset_type_id = [402003]
    # fund_id 为符合条件的基金名单
    ret = list()
    for i in operate_mode_id:
        for j in underlying_asset_type_id:
            tmp = fund_find(fund_stime, i, j)
            ret.append(tmp)
    fund_id = pd.concat(ret)
    fund_id['end_date'] = fund_id['end_date'].fillna(1)
    fund_id = fund_id[fund_id['end_date'] == 1]
    
    fund_lst = fund_id.main_code.tolist()
    ret = list()
    for fd in fund_lst:
        print(fd + ' is cal')
        q = query(finance.FUND_PORTFOLIO.code,
                  finance.FUND_PORTFOLIO.period_end,
                  finance.FUND_PORTFOLIO.fixed_income_rate).filter(finance.FUND_PORTFOLIO.code==fd).order_by(finance.FUND_PORTFOLIO.period_end.desc()).limit(12)
        ss = finance.run_query(q)
        ret.append(ss)
    fund_hold_ratio = pd.concat(ret)
    out = fund_hold_ratio.groupby(['code'])['fixed_income_rate'].mean().reset_index()
    return out

def cal_stock_hold_ratio():
    today = datetime.datetime.today()
    today = str(today)[:10]
    period = 36
    fund_stime = str(datetime.datetime.today() - relativedelta(months=period))[:10]  # 基金池开始时间

    operate_mode_id = [401001, 401006]
    underlying_asset_type_id = [402001]
    # fund_id 为符合条件的基金名单
    ret = list()
    for i in operate_mode_id:
        for j in underlying_asset_type_id:
            tmp = fund_find(fund_stime, i, j)
            ret.append(tmp)
    fund_id = pd.concat(ret)
    fund_id['end_date'] = fund_id['end_date'].fillna(1)
    fund_id = fund_id[fund_id['end_date'] == 1]
    
    fund_lst = fund_id.main_code.tolist()
    ret = list()
    for fd in fund_lst:
        print(fd + ' is cal')
        q = query(finance.FUND_PORTFOLIO.code,
                  finance.FUND_PORTFOLIO.period_end,
                  finance.FUND_PORTFOLIO.stock_rate).filter(finance.FUND_PORTFOLIO.code==fd).order_by(finance.FUND_PORTFOLIO.period_end.desc()).limit(12)
        ss = finance.run_query(q)
        ret.append(ss)
    fund_hold_ratio = pd.concat(ret)
    out = fund_hold_ratio.groupby(['code'])['stock_rate'].mean().reset_index()
    return out

def fund_type_label():
    hybrid_r = cal_hybrid_hold_ratio()
    hybrid_r['label'] = hybrid_r.stock_rate.apply(lambda s : '偏股混合型' if s > 70 
                                                  else '偏债混合型' if s < 30 else '标准混合型')
    
    bond_r = cal_bond_hold_ratio()
    bond_r['label'] = bond_r.fixed_income_rate.apply(lambda s : '纯债型' if s >= 90 
                                                  else '进取债型')
    stock_r = cal_stock_hold_ratio()
    stock_r = stock_r.query("stock_rate > 70")
    stock_r['label'] = '股票型'
    
    # 合成基金类型标签文件
    type_ = list()
    type_.append(hybrid_r[['code','label']])
    type_.append(bond_r[['code','label']])
    type_.append(stock_r[['code','label']])
    fund_type = pd.concat(type_)
    fund_type.to_csv('fund_type.csv',encoding='gbk')
    return "sussess"

def fund_hold_(code, date):
    q = query(finance.FUND_PORTFOLIO_STOCK.code, 
              finance.FUND_PORTFOLIO_STOCK.period_end,
              finance.FUND_PORTFOLIO_STOCK.symbol,
              finance.FUND_PORTFOLIO_STOCK.name,
              finance.FUND_PORTFOLIO_STOCK.proportion).filter(finance.FUND_PORTFOLIO_STOCK.code==code,
                                                              finance.FUND_PORTFOLIO_STOCK.period_end==date)
    df = finance.run_query(q)
    df.drop_duplicates(inplace=True)
    df['stock_market'] = df.symbol.apply(lambda s: len(s))
    df['first_code'] = df.symbol.apply(lambda s: str(s)[0])
    df = df.query("stock_market == 6")
    df = df[df.first_code.isin(['0','3','6'])]   
    df = df.sort_values(by='proportion').tail(10)
    return df

def get_industry_ratio(code, pub_day):
    stock_tmp = fund_hold_(code, pub_day)
    if len(stock_tmp) > 0:
        try:
            stock_tmp.drop_duplicates(inplace=True)
            stock_tmp['Industry'] = stock_tmp.symbol.apply(normalize_code)
            stock_tmp['Industry'] = stock_tmp.Industry.apply(check_industry)
            stock_ind_tmp = stock_tmp[['Industry', 'proportion']]
            if stock_ind_tmp['proportion'].sum() > 0:
                stock_ind_tmp['proportion'] = stock_ind_tmp['proportion'] / stock_ind_tmp['proportion'].sum()
                ind_ratio = stock_ind_tmp.groupby('Industry')['proportion'].sum().reset_index()
                ind_ratio['InnerCode'] = code
                ind_ratio['Date'] = pub_day
                return ind_ratio
            else:
                return []
        except:
            return []           
    else:
        return []
        

def fund_industry_label(pub_day):
    today = datetime.datetime.today()
    today = str(today)[:10]
    period = 36
    fund_stime = str(datetime.datetime.today() - relativedelta(months=period))[:10]  # 基金池开始时间
    # 行业名称
    indust_info = get_industries(name='sw_l1', date=None).reset_index().rename(columns={'index':'industry_code'})
    indust_info =indust_info[['industry_code', 'name']]
    indust_info.columns = ['Industry', 'indust_name'] 
    # 常量
    operate_mode_id = [401001, 401006]
    underlying_asset_type_id = [402001,402004]
    # fund_id 为符合条件的基金名单
    ret = list()
    for i in operate_mode_id:
        for j in underlying_asset_type_id:
            tmp = fund_find(fund_stime, i, j)
            ret.append(tmp)
    fund_id = pd.concat(ret)
    fund_id['end_date'] = fund_id['end_date'].fillna(1)
    fund_id = fund_id[fund_id['end_date'] == 1]
    
    fund_lst = fund_id.main_code.tolist()
    ret = list()
    for fd in fund_lst:
        print(fd + ' is cal')
        ss = get_industry_ratio(fd,pub_day)
        if len(ss) > 0:
            ret.append(ss)
    fund_hold_ratio = pd.concat(ret)
    
    ret_result = list()
    for idx,group in fund_hold_ratio.groupby('InnerCode'):
        tmp = group.sort_values(by='proportion',ascending=False)
        ret_result.append(tmp.head(1))
    fund_ind_ratio = pd.concat(ret_result)
    fund_ind_ratio = fund_ind_ratio.merge(indust_info)
    fund_ind_ratio.to_csv('fund_industry.csv',encoding='gbk')
    return "sussess"
    
    
if __name__ == '__main__':
    pub_day = '2020-06-30'
    # fund_type_label()
    # fund_industry_label(pub_day)
    
    


    

