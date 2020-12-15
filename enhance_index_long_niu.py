# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 19:35:44 2020
基于沪深300指数增强优选下的白马股优选策略
@author: Administrator
"""

from __future__ import division
import pandas as pd
import numpy as np
from datetime import datetime
from jqdatasdk import *
auth('18610039264','zg19491001')


# 提取基金净值数据
def fund_value(start_day,code):
      q=query(finance.FUND_NET_VALUE.code,
              finance.FUND_NET_VALUE.day,
              finance.FUND_NET_VALUE.sum_value,
              finance.FUND_NET_VALUE.refactor_net_value).filter(finance.FUND_NET_VALUE.code==code,
              finance.FUND_NET_VALUE.day> start_day)
      df=finance.run_query(q)
      return(df)

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

# 信息比率
def ir(netlist1, netlist2):
    asset_return = pd.Series(netlist1)
    index_return = pd.Series(netlist2)
    multiplier = 252
    
    if asset_return is not None and index_return is not None:

        active_return = asset_return - index_return
        tracking_error = (active_return.std(ddof = 1))* np.sqrt(multiplier)

        asset_annualized_return = multiplier * asset_return.mean()
        index_annualized_return = multiplier * index_return.mean()

        information_ratio = (asset_annualized_return - index_annualized_return)/tracking_error

    else:
        information_ratio = np.nan

    return information_ratio, tracking_error

def positive_ratio(r):
    p = [i for i in r if i > 0]

    return len(p) / len(r)

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



# 根据评分和分为点进行基金评级
def rank_stars(rejectdata, col_name, list_):
    # 进行星级评价
    num = 0
    for idx, group in rejectdata.groupby('M_O_Date'):
        tmp = group.copy()
        one_point = np.quantile(tmp[col_name], list_[0], interpolation='linear')
        two_point = np.quantile(tmp[col_name], list_[1], interpolation='linear')
        three_point = np.quantile(tmp[col_name], list_[2], interpolation='linear')
        four_point = np.quantile(tmp[col_name], list_[3], interpolation='linear')
        f = lambda s: 1 if s < one_point else 2 if (s >= one_point and s < two_point) else 3 if (
                    s >= two_point and s < three_point) else 4 if (s >= three_point and s < four_point) else 5
        tmp['stars'] = tmp[col_name].apply(f)
        if num == 0:
            result = tmp
        else:
            result = pd.concat([result, tmp])
        num += 1
    return result

# 列去极值
def filter_extreme_3sigma(series,n=3): #3 sigma
  mean = series.mean()
  std = series.std()
  max_range = mean + n*std
  min_range = mean - n*std
  return np.clip(series,min_range,max_range)

# 指标集合去极值
def no_extreme(df):
    ret = list()
    for idx, group_ in df.groupby('M_O_Date'):
        print(idx)
        tmp = group_.copy()
        for i in range(2,tmp.shape[1]):
            tmp.iloc[:,i] = filter_extreme_3sigma(tmp.iloc[:,i])
        ret.append(tmp)
    rank_and_indicator_no_extrema = pd.concat(ret)
    return rank_and_indicator_no_extrema

def fund_hold(code, date):
    q = query(finance.FUND_PORTFOLIO_STOCK.code, 
              finance.FUND_PORTFOLIO_STOCK.period_end,
              finance.FUND_PORTFOLIO_STOCK.symbol,
              finance.FUND_PORTFOLIO_STOCK.name,
              finance.FUND_PORTFOLIO_STOCK.proportion).filter(finance.FUND_PORTFOLIO_STOCK.code==code,
                                                              finance.FUND_PORTFOLIO_STOCK.period_end==date)
    df = finance.run_query(q)
    return df
    
if __name__ == '__main__':
    # 设置变量
    benchmark_code = '000300.XSHG'
    pub_date = '2019-12-31'
    # 数据加载区间
    sdate = '2010-01-01'
    edate = '2020-04-01'
    
    # 读取沪深300指数增强基金列表
    code_df = pd.read_csv('hs300_index_enhancement.csv')
    code_df.code = code_df.code.apply(lambda s: s[:6])
    code_lst = code_df.code.to_list() 
    # code_lst = ['000410', '519732', '519697', '000390', '200016']

    
    # 读取基金净值数据
    ret = list()
    for i in code_lst:
        print(i + ' is caluating of values')
        tmp = fund_value('2010-01-01', i)
        tmp['chg'] = tmp.sum_value.diff(1) / tmp.sum_value
        ret.append(tmp)
    value_data = pd.concat(ret)
    value_data.dropna(inplace=True)
    value_data.day = value_data.day.apply(lambda s:str(s)[:10])
    
    # 读取业绩基准数据
    benchmark = stock_price('000300.XSHG','1d', sdate, edate)
      
    benchmarknet = benchmark[['tradedate','close']].rename(columns=\
                            {'tradedate':'day','close' : 'b_mark'})
    
    benchmarknet['b_mark_chg'] = benchmarknet.b_mark.diff(1) / benchmarknet.b_mark
    benchmarknet.dropna(inplace=True)
    benchmarknet.day = benchmarknet.day.apply(lambda s:str(s)[:10])
    # 数据拼接
    df = value_data[['code', 'day', 'chg']].merge(benchmarknet[['day', 'b_mark_chg']])
    df['ex_chg'] = df['chg'] - df['b_mark_chg']
    
    # 计算评级指标
    # 分别计算日平均超额收益率，日超额收益胜率，日跟踪误差, 信息比率
    ret = list()
    for idx, group in df.groupby('code'):
        print(idx + ' is cal of indicators')
        res = list()
        tmp = group.copy()
        tmp = tmp.sort_values(by='day')
        chg_lst = tmp.chg.tolist()
        b_mark_lst = tmp.b_mark_chg.tolist()
        ex_chg_list = tmp.ex_chg.tolist()
        IR, trace_error = ir(chg_lst, b_mark_lst)
        ex_chg_mean = np.mean(ex_chg_list)*100
        p_ratio = positive_ratio(ex_chg_list)
        res.append(idx)
        res.append(ex_chg_mean)
        res.append(p_ratio)
        res.append(IR)
        res.append(trace_error)
        ret.append(res)
    fund_indicator = pd.DataFrame(ret, columns=['code', 'ex_chg_mean', 'p_ratio', 'ir', 'trace_error'])
    # 数据去极值，标准化  
    fund_indicator.insert(1,'M_O_Date', edate[:7])
    fund_indicator = no_extreme(fund_indicator)
    fund_indicator.iloc[:,2:] = fund_indicator.iloc[:,2:].apply(lambda x: (x - x.mean()) / x.std(ddof = 1))
    # topsis评级
    zb_lst = ['ex_chg_mean', 'p_ratio', 'ir', 'trace_error']
    direc = [1, 1, 1, -1]
    w = [2, 1, 2, 1]
    break_point_lst = [0.15, 0.70, 0.85, 0.95]  # 分位点
    topsis_score = topsis(fund_indicator, zb_lst, direc, w)
    funds_topsis_rank = rank_stars(topsis_score, 'topsis', break_point_lst)
    # 筛选
    funds_selected = funds_topsis_rank[funds_topsis_rank['stars'] >= 4]
    fund_selected_lst = funds_selected.code.tolist()
    # 提取基金持仓信息
    ret = list()
    for i in fund_selected_lst:
        print(i + ' is cal of holding')
        ret.append(fund_hold(i, pub_date))
    fund_hold = pd.concat(ret)
    
    # 持仓分析
    fund_hold_ratio = fund_hold.groupby(['symbol', 'name'])['proportion'].mean().reset_index()

    fund_hold_counts = fund_hold.groupby(['symbol', 'name'])['proportion'].count().reset_index()
    fund_hold_counts = fund_hold_counts.rename(columns={'proportion':'counts'})
    fund_hold_stat = fund_hold_ratio.merge(fund_hold_counts,on=['symbol', 'name'])
    # 获取参考标的权重
    index_weight = get_index_weights(index_id=benchmark_code, date=pub_date).reset_index()
    fund_hold_stat.symbol = fund_hold_stat.symbol.apply(normalize_code)
    fund_hold_stat = fund_hold_stat.rename(columns={'symbol':'code'})
    stat = fund_hold_stat.merge(index_weight[['code','weight']])
    stat['diff'] = stat['proportion'] - stat['weight']
    stat = stat.sort_values(by='diff',ascending=False)