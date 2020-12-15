# -*- coding: utf-8 -*-
"""
Created on Fri May 15 16:12:08 2020
识别基金屡创新高的能力
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
from hybrid_hold_stocks_ratio import *

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
        .rename(columns={'index': 'tradedate'})
    temp['stockcode'] = sec
    return temp


def price_to_rev(netlist):
    r = list()
    for i in range(1, len(netlist)):
        r.append(math.log(netlist[i] / netlist[i - 1]))
    return r


def get_dynamic_drawdown(asset_return):

    gr = asset_return[['log_return', 'EndDate']]
    gr = gr.set_index('EndDate')
    gr = gr.dropna(axis=0, how='any')
    running_max = np.maximum.accumulate(gr.cumsum())
    underwater = gr.cumsum() - running_max
    underwater = np.exp(underwater) - 1
    underwater = underwater.reset_index()
    underwater = underwater.rename(columns={'log_return': 'drawdown'})
    return underwater

def cal_dynamic_drawdown_ind(code, df_rev):
    tmp = df_rev.query("code == '%s'"%code)
    ss = get_dynamic_drawdown(tmp)
    # 动态回撤平均归零时间，最大归零时间
    back_zero_time = list()
    is_zero = 1
    is_first = 1
    num = 0
    for i in ss['drawdown']:
        num += 1
        if (i !=0) & (is_first == 1):
            is_first = 0
            is_zero = 0
            counts = list()
            counts.append(i)
            continue
            
        if (i !=0) & (is_zero==1):
            is_zero = 0
            counts = list()
            counts.append(i)
            continue
        if (i != 0) & (is_zero == 0):
            counts.append(i)
        if ((i == 0) & (is_zero == 0)) | ((i != 0) & (num == len(ss))):
            back_zero_time.append(counts)
            is_zero = 1
    # 统计各种指标
    draw_num = len(back_zero_time)
    draw_days = [ len(d) for d in back_zero_time]
    draw_days_max = max(draw_days)
    draw_days_mean = np.mean(draw_days)
    draw = [min(s) for s in back_zero_time]        
    draw_max = min(draw) 
    draw_mean = np.mean(draw)
    return [draw_days_max, draw_days_mean, draw_max, draw_mean]

def topsis(df, zb_lst, direc, w):
    temp = df.copy()
    zb_list = zb_lst
    tmp = temp[zb_list]
    tmp1 = (tmp - tmp.min()) / (tmp.max() - tmp.min())
    g = list()
    b = list()
    for i in range(len(direc)):
        if direc[i] > 0:
            g.append(tmp1.iloc[:, i].max())
            b.append(tmp1.iloc[:, i].min())
        else:
            g.append(tmp1.iloc[:, i].min())
            b.append(tmp1.iloc[:, i].max())
    g_d = pd.Series(g, index=zb_list)
    b_d = pd.Series(b, index=zb_list)
    tmp2 = tmp1.copy()
    G_tmp = tmp1 - g_d

    tmp2['G'] = G_tmp.apply(lambda x: (x.multiply(x)).multiply(np.array(w)).sum() ** 0.5, axis=1)

    B_tmp = tmp1 - b_d
    tmp2['B'] = B_tmp.apply(lambda x: (x.multiply(x)).multiply(np.array(w)).sum() ** 0.5, axis=1)
    temp['topsis'] = tmp2['B'] / (tmp2['G'] + tmp2['B'])
    return temp


if __name__ == '__main__':
    
    today = datetime.datetime.today()
    today = str(today)[:10]
    period = 62
    fund_stime = str(datetime.datetime.today() - relativedelta(months=period))[:10]  # 基金池开始时间

    operate_mode_id = [401001, 401003, 401006]
    #underlying_asset_type_id = [402001, 402003, 402004]
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
    
    # 计算对数收益率序列
    values = df[['code', 'day' ,'sum_value']]
    ret = list()
    for idx, group_ in df.groupby("code"):
        print(idx)
        tmp = group_.copy()
        tmp.dropna(subset=['sum_value'], inplace=True)
        if len(tmp) > period * 20 * 0.8:
            tmp = tmp.sort_values(by='day')
            tmp['log_return'] = np.log(tmp.sum_value)
            tmp['log_return'] = tmp['log_return'].diff()
            ret.append(tmp)
    df_rev = pd.concat(ret)
    df_rev = df_rev.rename(columns={'day' : 'EndDate'})
    # check = df_rev.query("code == '161811'")
    fund_lst = df_rev.code.drop_duplicates().tolist()
    # 计算动态回撤分析指标
    ret =list()
    for fc in fund_lst:
        res = list()
        print(fc)
        ss = cal_dynamic_drawdown_ind(fc, df_rev)
        res.append(fc)
        res.append(ss[0]) 
        res.append(ss[1])
        res.append(ss[2])
        res.append(ss[3])
        ret.append(res)
    dynamic_draw_any = pd.DataFrame(ret, columns=['code', 'draw_days_max', 'draw_days_mean',
                                                'draw_max', 'draw_mean'])
        
    # 计算动态回撤率衍生指标
    info_draw_anly = fund_id[['main_code', 'name', 'underlying_asset_type']]
    info_draw_anly = info_draw_anly.rename(columns={'main_code':'code'})
    info_draw_anly = info_draw_anly.merge(dynamic_draw_any, on='code')
    ret = list()
    for idx, group_ in info_draw_anly.groupby('underlying_asset_type'):
        tmp = group_.copy()
        
        zb_lst = ['draw_days_max', 'draw_days_mean','draw_max', 'draw_mean']
        direc = [-1, -1, 1, 1]
        w = [1, 1, 1, 1]
        df = topsis(tmp, zb_lst, direc, w)
        ret.append(df)
    fund_analysis = pd.concat(ret)
    
    # 混合型标签
    hy_rd = cal_hybrid_hold_ratio()
    fund_analysis_ = fund_analysis.merge(hy_rd, on='code', how='left')
    fund_analysis_.to_excel('dy_draw.xls',encoding='gbk')
