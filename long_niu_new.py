# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 15:07:40 2020
新白马股筛选
@author: Administrator
"""

from __future__ import division
import pandas as pd
import numpy as np
import statsmodels.api as sm
import math
from datetime import datetime
from dateutil.relativedelta import relativedelta
# import talib
from jqdatasdk import *

auth('18610039264', 'zg19491001')


# 获取价格
def stock_price(sec, period, sday, eday):
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


# 累计收益率
def cumror(netlist):
    '''
    :param netlist:净值曲线
    :return: 累计收益
    '''
    return math.pow(netlist[-1] / netlist[0], 1) - 1


# 累计超额收益率
def ex_cumror(netlist1, netlist2):
    cumror1 = cumror(netlist1)
    cumror2 = cumror(netlist2)

    return cumror1 - cumror2


# 年化收益率
def annror(netlist, days):
    '''
    :param netlist:净值曲线
    :return: 年化收益
    '''
    annror_ = (1 + cumror(netlist)) ** (1 / (days / 365)) - 1
    return annror_


# 超额年化收益率
def ex_annror(netlist1, netlist2, days):
    ex_annror_1 = annror(netlist1, days)
    ex_annror_2 = annror(netlist2, days)
    return ex_annror_1 - ex_annror_2


# 计算峰度
def kurtosis(netlist):
    r = price_to_rev(netlist)
    s = pd.Series(r)

    return s.kurt()


# 超额收益率的峰度
def ex_kurtosis(netlist1, netlist2):
    r1 = price_to_rev(netlist1)
    r2 = price_to_rev(netlist2)
    ex_r = list()
    for i in range(len(r1)):
        ex_r.append(r1[i] - r2[i])
    s = pd.Series(ex_r)

    return s.kurt()


# 计算偏度
def skewness(netlist):
    r = price_to_rev(netlist)
    s = pd.Series(r)

    return s.skew()


# 超额收益率的偏度
def ex_skewness(netlist1, netlist2):
    r1 = price_to_rev(netlist1)
    r2 = price_to_rev(netlist2)
    ex_r = list()
    for i in range(len(r1)):
        ex_r.append(r1[i] - r2[i])
    s = pd.Series(ex_r)

    return s.skew()


# 计算正收益率占比
def positive_ratio(netlist):
    r = price_to_rev(netlist)
    p = [i for i in r if i > 0]

    return len(p) / len(r)


# 计算超额收益率正收益占比
def ex_positive_ratio(netlist1, netlist2):
    r1 = price_to_rev(netlist1)
    r2 = price_to_rev(netlist2)
    ex_r = list()
    for i in range(len(r1)):
        ex_r.append(r1[i] - r2[i])

    p = [i for i in ex_r if i > 0]

    return len(p) / len(ex_r)


# 计算收益率年化波动率：
def volatility(netlist):
    r = price_to_rev(netlist)

    return np.std(r, ddof=1) * math.pow(252, 0.5)


# 计算超额收益率年化波动率
def ex_volatility(netlist1, netlist2):
    r1 = price_to_rev(netlist1)
    r2 = price_to_rev(netlist2)
    ex_r = list()
    for i in range(len(r1)):
        ex_r.append(r1[i] - r2[i])

    return np.std(ex_r, ddof=1) * math.pow(252, 0.5)


# 最大回撤率
def maxretrace(netlist):
    r = price_to_rev(netlist)
    asset_return = pd.Series(r)
    if len(asset_return) > 1:
        running_max = np.maximum.accumulate(asset_return.cumsum())
        underwater = asset_return.cumsum() - running_max
        underwater = np.exp(underwater) - 1
        mdd = -underwater.min()
    else:
        mdd = np.nan
    return mdd


# 高水位回撤率
def avgretrace(netlist):
    '''
    :param list:netlist
    :return: 平均回撤率
    '''
    everyRetrace = list()
    for i in range(len(netlist) - 1):
        t = netlist[:(len(netlist) - i)]
        t_max = max(t)
        retrace = ((t_max - t[-1]) / t_max) * (0.9 ** (i))
        everyRetrace.append(retrace)

    return (sum(everyRetrace))


# 基于历史法计算Var 
def var(netlist, a=0.01):
    '''
    :param list:netlist
    :return: 平均回撤率
    '''
    # from scipy.stats import norm

    r = price_to_rev(netlist)
    r_s = pd.Series(r)
    # r_s_p = r_s.rolling(period).apply(np.sum, raw=True)
    r_s = r_s.dropna()
    var = np.quantile(r_s, a, interpolation='linear')

    return (var)


def down_std(netlist):
    multiplier = 252
    r = price_to_rev(netlist)
    asset_return = pd.Series(r)

    target_return = 0.03 / multiplier

    if asset_return is not None:

        downside_return = asset_return - target_return
        downside_return[downside_return > 0] = 0
        downside_volatility = downside_return.std(ddof=1) * np.sqrt(multiplier)
    else:
        downside_volatility = np.nan

    return downside_volatility


def ex_down_std(netlist1, netlist2):
    r1 = price_to_rev(netlist1)
    asset_return = pd.Series(r1)
    r2 = price_to_rev(netlist2)
    bemark = pd.Series(r2)
    multiplier = 252

    if asset_return is not None:

        downside_return = asset_return - bemark
        downside_return[downside_return > 0] = 0
        downside_volatility = downside_return.std(ddof=1) * np.sqrt(multiplier)
    else:
        downside_volatility = np.nan

    return downside_volatility


# 收益风险指标
# 计算alpha，beta
def alpha_and_beta(netlist1, netlist2):
    r1 = price_to_rev(netlist1)
    asset_return = pd.Series(r1)
    r2 = price_to_rev(netlist2)
    index_return = pd.Series(r2)
    multiplier = 252
    rf = 0.03 / multiplier
    if asset_return is not None and index_return is not None:
        y = asset_return - rf
        x = index_return - rf
        x = sm.add_constant(x)
        model = sm.OLS(y, x).fit()
        alpha = model.params.iloc[0]
        alpha = multiplier * alpha
        if len(y) > 1:
            beta = model.params.iloc[1]
        else:
            beta = np.nan
    else:
        alpha = np.nan
        beta = np.nan

    return alpha, beta


# 夏普比率
def yearsharpRatio(netlist):
    r = price_to_rev(netlist)
    asset_return = pd.Series(r)
    multiplier = 252

    if asset_return is not None:

        annualized_return = multiplier * asset_return.mean()
        annualized_vol = asset_return.std(ddof=1) * np.sqrt(multiplier)

        sharpe_ratio = (annualized_return - 0.03) / annualized_vol

    else:
        sharpe_ratio = np.nan

    return sharpe_ratio


# 索提诺比率
def sortinoratio(netlist):
    r = price_to_rev(netlist)
    asset_return = pd.Series(r)
    multiplier = 252

    if asset_return is not None:

        target_return = 0.03 / multiplier

        downside_return = asset_return - target_return
        downside_return[downside_return > 0] = 0
        downside_volatility = downside_return.std(ddof=1) * np.sqrt(multiplier)
        annualized_return = multiplier * asset_return.mean()
        sortino_ratio = (annualized_return - 0.03) / downside_volatility

    else:
        sortino_ratio = np.nan

    return sortino_ratio


# 信息比率
def ir(netlist1, netlist2):
    r1 = price_to_rev(netlist1)
    asset_return = pd.Series(r1)
    r2 = price_to_rev(netlist2)
    index_return = pd.Series(r2)
    multiplier = 252

    if asset_return is not None and index_return is not None:

        active_return = asset_return - index_return
        tracking_error = (active_return.std(ddof=1)) * np.sqrt(multiplier)

        asset_annualized_return = multiplier * asset_return.mean()
        index_annualized_return = multiplier * index_return.mean()

        information_ratio = (asset_annualized_return - index_annualized_return) / tracking_error

    else:
        information_ratio = np.nan

    return information_ratio


# 收益回撤比
def rd(netlist, days):
    return (annror(netlist, days) / maxretrace(netlist))


# 特雷诺比率
def treynor(netlist1, netlist2):
    r1 = price_to_rev(netlist1)
    asset_return = pd.Series(r1)
    r2 = price_to_rev(netlist2)
    index_return = pd.Series(r2)
    multiplier = 252
    rf = 0.03 / multiplier
    if asset_return is not None and index_return is not None:
        y = asset_return - rf
        x = index_return - rf
        x = sm.add_constant(x)
        model = sm.OLS(y, x).fit()
        # alpha = model.params[0]
        if len(y) > 1:
            beta = model.params.iloc[1]
        else:
            beta = np.nan

        annualized_return = multiplier * asset_return.mean()

        treynor_ratio = (annualized_return - 0.03) / beta

    else:
        treynor_ratio = np.nan

    return treynor_ratio


# 获取当前全部股票信息
def all_stocks(today):
    """
    输出股票代码，股票名称，股票简称，上市时间等主要字段
    输出格式 dataframe
    """
    return get_all_securities(types=[], date=today).reset_index().rename(columns={'index': 'stockcode'})


# 将日频数据转化为月频数据
def data_monthly(df):
    ret = list()
    for idx, group_ in df.groupby(['stockcode', 'M_O_Date']):
        ret.append(group_.head(1).copy())
    return pd.concat(ret)


# 指标切片
def data_period_split(df, indicator_lst):
    num = 0
    indictors_many_period = list()
    for idx, group in df.groupby('period'):
        col = [i + '_' + str(idx) for i in indicator_lst]
        indictors_many_period.extend(col)
        col_p = ['stockcode', 'tradedate']
        col_p.extend(col)
        tmp = group.copy()
        tmp = tmp.iloc[:, :-1]
        tmp.columns = col_p
        if num == 0:
            ret = tmp.copy()
        else:
            ret = pd.merge(ret, tmp.copy(), on=['stockcode', 'tradedate'], how='inner')
        num += 1
    return ret, indictors_many_period


# topsis评分
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


# 生成全截面的topsis评分
def generate_topsis_score(df, col, zb_lst, direc, w):
    fundrank_input = df[col]
    fundrank_input = fundrank_input.dropna()
    num = 0
    for idx, group in fundrank_input.groupby('M_O_Date'):
        print(idx)
        tmp = group.copy()
        tmp = topsis(tmp, zb_lst
                     , direc, w)
        if num == 0:
            out = tmp.copy()
        else:
            out = pd.concat([out, tmp.copy()])
        num = num + 1
    # out.to_csv('result/fund_rank_topsis.csv')
    return out


# 根据评分和分为点进行基金评级
def rank_stars(rejectdata, col_name, list_=[0.15, 0.7, 0.85, 0.95]):
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
def filter_extreme_MAD(series, n=7):  # 3 MAD
    median = series.quantile(0.5)
    new_median = ((series - median).abs()).quantile(0.50)
    max_range = median + n * new_median
    min_range = median - n * new_median
    return np.clip(series, min_range, max_range)


# 指标集合去极值
def no_extreme(df, col):
    tmp = df.copy()
    for i in col:
        tmp.loc[:, i] = filter_extreme_MAD(tmp.loc[:, i])
    return tmp


def check_industry(code, indust_classify_code='sw_l1'):
    try:
        return get_industry(code)[code][indust_classify_code]['industry_code']
    except:
        return 0


if __name__ == '__main__':
    # 行情数据起始和中止点

    # end = str(datetime.today())[:10]
    end = '2020-05-20'
    start = str(datetime.today() - relativedelta(months=65))[:10]
    # 业绩基准
    benchmark_code = '000300.XSHG'
    # 全部股票池
    allstock = all_stocks(end)
    stocks = allstock.copy()
    stocks['duration'] = stocks.start_date.apply(lambda s: (datetime.today() - s).days)
    stocks['st'] = stocks.display_name.apply(lambda s: 'ST' in s)
    stocks = stocks.query("duration > 1800 & st == False")
    allstock_lst = stocks.stockcode.tolist()
    # 业绩基准行情
    b_mark = stock_price(benchmark_code, '1d', start, end)

    # # 行情数据
    # price_data = list()
    # for i in allstock_lst:
    #     print(i + ' is caluating for get_price')
    #     tmp = stock_price(i, '1d', start, end)
    #     price_data.append(tmp)
    # price_data_df = pd.concat(price_data)
    # price_data_df.to_csv('long_niu_v2_stockdata.csv')
    # price_data_df['M_O_Date'] = price_data_df['tradedate'].apply(lambda x: x.strftime('%Y-%m'))

    # # 行情数据转月频
    # price_data_df_tmp = price_data_df[ price_data_df['M_O_Date'] == end[:7]]
    # mo = data_monthly(price_data_df_tmp)
    # month_period = [6, 12, 24, 36, 60]
    # mo_split = mo.dropna()

    # # 计算各截面的评价指标
    # res = list()
    # for idx, group in mo_split.groupby('stockcode'):
    #     print(idx)
    #     # if idx not in ['000001.XSHE']:
    #     #     break            
    #     tmp = group.copy()
    #     for p in month_period:
    #         print(p)
    #         tmp['start_time'] = tmp.tradedate.apply(lambda s: s - relativedelta(months=p))
    #         fund_net = price_data_df[price_data_df['stockcode'] == idx]
    #         for itdx, row in tmp.iterrows():
    #             ret = list()
    #             s_time = row['start_time']
    #             e_time = row['tradedate']
    #             day_count = (e_time - s_time).days
    #             tmp_net = fund_net[(fund_net['tradedate'] >= s_time)
    #                                 & (fund_net['tradedate'] < e_time)]
    #             benchmark_net = b_mark[(b_mark['tradedate'] >= s_time)
    #                                     & (b_mark['tradedate'] < e_time)]
    #             fund = tmp_net[['tradedate', 'close']]
    #             fund.columns = ['tradedate', 's_close']
    #             benchmark = benchmark_net[['tradedate', 'close']]
    #             benchmark.columns = ['tradedate', 'b_close']
    #             fund_benchmark = pd.merge(fund, benchmark, on='tradedate'
    #                                       , how='inner')
    #             if len(tmp_net) >= p * 20 * 0.7:
    #                 try:
    #                     fund_net_list = fund_benchmark.s_close.tolist()
    #                     benchmark_net_list = fund_benchmark.b_close.tolist()
    #                     # InnerCode = idx
    #                     # EndDate = e_time
    #                     cumror_ = cumror(fund_net_list)
    #                     ex_cumror_ = ex_cumror(fund_net_list, benchmark_net_list)
    #                     annror_ = annror(fund_net_list, day_count)
    #                     ex_annror_ = ex_annror(fund_net_list, benchmark_net_list, day_count)
    #                     kurtosis_ = kurtosis(fund_net_list)
    #                     ex_kurtosis_ = ex_kurtosis(fund_net_list, benchmark_net_list)
    #                     skewness_ = skewness(fund_net_list)
    #                     ex_skewness_ = ex_skewness(fund_net_list, benchmark_net_list)
    #                     positive_ratio_ = positive_ratio(fund_net_list)
    #                     ex_positive_ratio_ = ex_positive_ratio(fund_net_list, benchmark_net_list)
    #                     volatility_ = volatility(fund_net_list)
    #                     ex_volatility_ = ex_volatility(fund_net_list, benchmark_net_list)
    #                     maxretrace_ = maxretrace(fund_net_list)
    #                     avgretrace_ = avgretrace(fund_net_list)
    #                     var_1 = var(fund_net_list)
    #                     # var_3 = var(fund_net_list, 3)
    #                     # var_5 = var(fund_net_list, 5)
    #                     down_risk_ = down_std(fund_net_list)
    #                     down_std_ = ex_down_std(fund_net_list, benchmark_net_list)
    #                     alpha_beta = alpha_and_beta(fund_net_list, benchmark_net_list)
    #                     yearsharpRatio_ = yearsharpRatio(fund_net_list)
    #                     sortinoratio_ = sortinoratio(fund_net_list)
    #                     ir_ = ir(fund_net_list, benchmark_net_list)
    #                     rd_ = rd(fund_net_list, day_count)
    #                     treynor_ = treynor(fund_net_list, benchmark_net_list)

    #                     ret.append(idx)
    #                     ret.append(e_time)
    #                     ret.append(cumror_)
    #                     ret.append(ex_cumror_)
    #                     ret.append(annror_)
    #                     ret.append(ex_annror_)
    #                     ret.append(kurtosis_)
    #                     ret.append(ex_kurtosis_)
    #                     ret.append(skewness_)
    #                     ret.append(ex_skewness_)
    #                     ret.append(positive_ratio_)
    #                     ret.append(ex_positive_ratio_)
    #                     ret.append(volatility_)
    #                     ret.append(ex_volatility_)
    #                     ret.append(maxretrace_)
    #                     ret.append(avgretrace_)
    #                     ret.append(var_1)
    #                     # ret.append(var_3)
    #                     # ret.append(var_5)
    #                     ret.append(down_risk_)
    #                     ret.append(down_std_)
    #                     ret.append(alpha_beta[0])
    #                     ret.append(alpha_beta[1])
    #                     ret.append(yearsharpRatio_)
    #                     ret.append(sortinoratio_)
    #                     ret.append(ir_)
    #                     ret.append(rd_)
    #                     ret.append(treynor_)
    #                     ret.append(p)
    #                     res.append(ret)

    #                 except:
    #                     ret.append(idx)
    #                     ret.append(e_time)
    #                     ret.append(np.nan)
    #                     ret.append(np.nan)
    #                     ret.append(np.nan)
    #                     ret.append(np.nan)
    #                     ret.append(np.nan)
    #                     ret.append(np.nan)
    #                     ret.append(np.nan)
    #                     ret.append(np.nan)
    #                     ret.append(np.nan)
    #                     ret.append(np.nan)
    #                     ret.append(np.nan)
    #                     ret.append(np.nan)
    #                     ret.append(np.nan)
    #                     ret.append(np.nan)
    #                     ret.append(np.nan)
    #                     ret.append(np.nan)
    #                     ret.append(np.nan)
    #                     ret.append(np.nan)
    #                     ret.append(np.nan)
    #                     ret.append(np.nan)
    #                     ret.append(np.nan)
    #                     ret.append(np.nan)
    #                     ret.append(np.nan)
    #                     ret.append(np.nan)
    #                     # ret.append(np.nan)
    #                     # ret.append(np.nan)
    #                     ret.append(p)
    #                     res.append(ret)


    #             else:
    #                 ret.append(idx)
    #                 ret.append(e_time)
    #                 ret.append(np.nan)
    #                 ret.append(np.nan)
    #                 ret.append(np.nan)
    #                 ret.append(np.nan)
    #                 ret.append(np.nan)
    #                 ret.append(np.nan)
    #                 ret.append(np.nan)
    #                 ret.append(np.nan)
    #                 ret.append(np.nan)
    #                 ret.append(np.nan)
    #                 ret.append(np.nan)
    #                 ret.append(np.nan)
    #                 ret.append(np.nan)
    #                 ret.append(np.nan)
    #                 ret.append(np.nan)
    #                 ret.append(np.nan)
    #                 ret.append(np.nan)
    #                 ret.append(np.nan)
    #                 ret.append(np.nan)
    #                 ret.append(np.nan)
    #                 ret.append(np.nan)
    #                 ret.append(np.nan)
    #                 ret.append(np.nan)
    #                 ret.append(np.nan)
    #                 # ret.append(np.nan)
    #                 # ret.append(np.nan)
    #                 ret.append(p)
    #                 res.append(ret)

    # result = pd.DataFrame(res)
    # result = result.dropna()
    # result.columns = ['stockcode', 'tradedate', 'cumror', 'ex_cumror'
    #     , 'annror', 'ex_annror', 'kurtosis', 'ex_kurtosis'
    #     , 'skewness', 'ex_skewness', 'positive_ratio'
    #     , 'ex_positive_ratio', 'volatitility'
    #     , 'ex_volatitility', 'maxretrace', 'avgretrace'
    #     , 'var_1', 'down_std'
    #     , 'ex_down_std', 'alpha', 'beta', 'yearsharp'
    #     , 'sortinoratio', 'ir', 'rd', 'treynor', 'period']
    
  
    # # result.to_csv('indicator_stock_v200509.csv')
    result = pd.read_csv('indicator_stock_v200509.csv', index_col=0)
    # 计算每个截面的topsis 
    INDICATORS = ['cumror', 'ex_cumror'
        , 'annror', 'ex_annror', 'kurtosis', 'ex_kurtosis'
        , 'skewness', 'ex_skewness', 'positive_ratio'
        , 'ex_positive_ratio', 'volatitility'
        , 'ex_volatitility', 'maxretrace', 'avgretrace'
        , 'var_1', 'down_std'
        , 'ex_down_std', 'alpha', 'beta', 'yearsharp'
        , 'sortinoratio', 'ir', 'rd', 'treynor']

    INDICATORS_AFTER_SELECT = ['ex_annror_36',
                                'alpha_36',
                                'ir_60',
                                'maxretrace_60',
                                'var_1_6',
                                'down_std_36',
                                'avgretrace_60',
                                'ir_36', 'sortinoratio_6', 'rd_24']

    direc = [1, 1, 1, -1, -1, -1, -1, 1, 1, 1]
    w = [1, 1, 1, 6, 1, 1, 2, 1, 1, 1]
    col = ['stockcode', 'tradedate', 'M_O_Date']
    col.extend(INDICATORS_AFTER_SELECT)
    break_point_lst = [0.15, 0.70, 0.85, 0.95]  # 评级标准-分位点设置
    # 读取个股评价指标数据
    df = result.copy()
    # 对指标按照滚动窗口期进行切分
    df, indicators = data_period_split(df, INDICATORS)
    df['M_O_Date'] = df.tradedate.apply(lambda s: str(s)[:7])
    # df.to_excel('stock_long_niu_0509_ind.xls', encoding='gbk')
    # 去极值，标准化
    df_after_select = no_extreme(df, indicators)
    
    # df_after_select.to_excel('stock_long_niu_0509_ind_1.xls', encoding='gbk')
    df_after_select.loc[:, indicators] = df_after_select.loc[:, indicators] \
                                            .apply(lambda x: (x - x.mean()) / x.std())


    # # 计算生成个股ＴＯＰＳＩＳ评分   
    funds_topsis_score = generate_topsis_score(df_after_select, col, INDICATORS_AFTER_SELECT, direc, w)
    # 基于评分给定个股评级
    funds_topsis_rank = rank_stars(funds_topsis_score, 'topsis', break_point_lst)
    # 贴股票信息 
    stock_info = allstock.iloc[:, :2]
    stock_rank = stock_info.merge(funds_topsis_rank, on='stockcode')
    stock_rank['indust'] = stock_rank.stockcode.apply(check_industry)
    
    indust_info = get_industries(name='sw_l1', date=None).reset_index().rename(columns={'index':'industry_code'}) 
    indust_info =indust_info[['industry_code', 'name']]
    indust_info.columns = ['indust', 'indust_name'] 
    stock_rank = stock_rank.merge(indust_info, on='indust')
    stock_rank.to_excel('stock_long_niu_0602.xls', encoding='gbk')
    
# =============================================================================
#  提取4星以上基金
# =============================================================================
    stock_rank = pd.read_excel('stock_long_niu_0602.xls', encoding='gbk',index_col=0)
    lfp_index_pool = stock_rank[['stockcode','stars','indust']]
    lfp_index_pool = lfp_index_pool.query("stars>=4")
    lfp_index_pool.indust = lfp_index_pool.indust.apply(lambda s: str(s))
    ret = list()
    for idx, group_ in lfp_index_pool.iterrows():
        if group_['indust'] in ['801710', '801720', '801890', '801730']:
            ret.append('大基建')
        elif group_['indust'] in ['801110', '801120', '801130','801140']:
            ret.append('大消费')
        elif group_['indust'] in ['801150']:
            ret.append('大健康')
        elif group_['indust'] in ['801780', '801790']:
            ret.append('大金融')
        elif group_['indust'] in ['801080', '801750', '801760', '801770']:
            ret.append('TMT')
        elif group_['indust'] in ['801020', '801030', '801040', '801050']:
            ret.append('大资源')
        elif group_['indust'] in ['801180']:
            ret.append('大地产')            
        else:
            ret.append('其他')
    lfp_index_pool['category'] = ret
    lfp_index_pool = lfp_index_pool[lfp_index_pool['category']!='其他']
    lfp_index_pool.to_csv("lfp_good_stocks.csv", encoding='gbk')      

        
