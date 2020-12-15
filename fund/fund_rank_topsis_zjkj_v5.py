#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 10:34:39 2019
基于智君科技工作内容，将有效指标作为入参
输出TOPSIS评分- 确定评级指标，去极值，标准化，
基于二级分类文件-输出一种评级，不分3年 5年- 不进行筛选，输出全部评级结果
topsis序逻辑
@author: lufeipeng
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


# 窗口切分
def data_period_split(df, indicator_lst):
    num = 0
    indictors_many_period = list()
    for idx, group in df.groupby('period'):
        col = [i + '_' + str(idx) for i in indicator_lst]
        indictors_many_period.extend(col)
        col_p = ['code', 'day']
        col_p.extend(col)
        tmp = group.copy()
        tmp = tmp.iloc[:, :-1]
        tmp.columns = col_p
        if num == 0:
            ret = tmp.copy()
        else:
            ret = pd.merge(ret, tmp.copy(), on=['code', 'day'], how='outer')
        num += 1
    return ret, indictors_many_period


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


def generate_topsis_score(df, col, zb_lst, direc, w):
    fundrank_input = df[col]
    fundrank_input = fundrank_input.dropna()
    ret = list()
    for idx, group in fundrank_input.groupby('M_O_Date'):
        print(idx)
        tmp = group.copy()
        tmp = topsis(tmp, zb_lst
                     , direc, w)
        ret.append(tmp)

    # out.to_csv('result/fund_rank_topsis.csv')
    return pd.concat(ret)


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

def get_dynamic_drawdown(asset_return):
    gr = asset_return[['log_return', 'day']]
    gr = gr.set_index('day')
    gr = gr.dropna(axis=0, how='any')
    running_max = np.maximum.accumulate(gr.cumsum())
    underwater = gr.cumsum() - running_max
    underwater = np.exp(underwater) - 1
    underwater = underwater.reset_index()
    underwater = underwater.rename(columns={'log_return': 'drawdown'})
    return underwater


def cal_dynamic_drawdown_ind(code, df_rev):
    tmp = df_rev.query("code == '%s'" % code)
    tmp = tmp[['day', 'log_return']].copy()
    ss = get_dynamic_drawdown(tmp)
    # 动态回撤平均归零时间，最大归零时间
    back_zero_time = list()
    is_zero = 1
    is_first = 1
    num = 0
    for i in ss['drawdown']:
        num += 1
        if (i != 0) & (is_first == 1):
            is_first = 0
            is_zero = 0
            counts = list()
            counts.append(i)
            continue

        if (i != 0) & (is_zero == 1):
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
    # draw_num = len(back_zero_time)
    draw_days = [len(d) for d in back_zero_time]
    draw_days_max = max(draw_days)
    draw_days_mean = np.mean(draw_days)
    draw = [min(s) for s in back_zero_time]
    draw_max = min(draw)
    draw_mean = np.mean(draw)
    return [draw_days_max, draw_days_mean, draw_max, draw_mean]

def str_code(df):
    df.code = df.code.apply(lambda s: str(s).zfill(6))
    return df
    


if __name__ == '__main__':

    today = datetime.datetime.today()
    today = str(today)[:10]
    fund_stime = str(datetime.datetime.today() - relativedelta(months=60))[:10]  # 基金池开始时间

    operate_mode_id = [401001, 401006]
    underlying_asset_type_id = [402001, 402003, 402004]

    # fund_id 为符合条件的基金名单
    ret = list()
    for i in operate_mode_id:
        for j in underlying_asset_type_id:
            tmp = fund_find(fund_stime, i, j)
            ret.append(tmp)
    fund_id = pd.concat(ret)

    fund_id['end_date'] = fund_id['end_date'].fillna(1)
    fund_id = fund_id[fund_id['end_date'] == 1]
    
    # 叠加二级分类
    fund_type2 =pd.read_csv('fund_type.csv',encoding='gbk',index_col=0)
    fund_type2 = fund_type2.rename(columns={'code':'main_code'})
    fund_type2.main_code = fund_type2.main_code.apply(lambda s: str(s).zfill(6))
    fund_id = fund_id.merge(fund_type2)
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
    df.to_csv('data/fund_value.csv')
    # df = pd.read_csv('data/fund_value.csv', encoding='gbk',index_col=0)

    benchmark = stock_price('000300.XSHG', '1d', fund_stime, today)

    benchmarknet = benchmark[['tradedate', 'close']].rename(columns=
                                                            {'tradedate': 'day', 'close': 'b_mark'})
    benchmarknet.day = benchmarknet.day.apply(lambda s: str(s)[:10])
    fund_value = df[['code', 'day', 'sum_value']]
    fund_value.day = fund_value.day.apply(lambda s: str(s))
    fund_value_bench = pd.merge(fund_value, benchmarknet, on='day', how='left')
    # 统计
    fund_value_bench = fund_value_bench.dropna()
    month_period = [6, 12, 24, 36, 60]
    res = list()
    for idx, group in fund_value_bench.groupby('code'):
        print(idx)
        tmp = group.copy()
        tmp = tmp.sort_values(by='day')
        for p in month_period:
            print(p)
            ret = list()
            now = datetime.datetime.today()
            date_s = str(now - relativedelta(months=p))[:10]
            day_count = (now - (now - relativedelta(months=p))).days
            fund_benchmark = tmp[tmp['day'] > date_s]
            if len(fund_benchmark) >= p * 20 * 0.9:
                try:
                    fund_net_list = fund_benchmark.sum_value.tolist()
                    benchmark_net_list = fund_benchmark.b_mark.tolist()
                    # InnerCode = idx
                    # EndDate = e_time
                    cumror_ = cumror(fund_net_list)
                    ex_cumror_ = ex_cumror(fund_net_list, benchmark_net_list)
                    annror_ = annror(fund_net_list, day_count)
                    ex_annror_ = ex_annror(fund_net_list, benchmark_net_list, day_count)
                    kurtosis_ = kurtosis(fund_net_list)
                    ex_kurtosis_ = ex_kurtosis(fund_net_list, benchmark_net_list)
                    skewness_ = skewness(fund_net_list)
                    ex_skewness_ = ex_skewness(fund_net_list, benchmark_net_list)
                    positive_ratio_ = positive_ratio(fund_net_list)
                    ex_positive_ratio_ = ex_positive_ratio(fund_net_list, benchmark_net_list)
                    volatility_ = volatility(fund_net_list)
                    ex_volatility_ = ex_volatility(fund_net_list, benchmark_net_list)
                    maxretrace_ = maxretrace(fund_net_list)
                    avgretrace_ = avgretrace(fund_net_list)
                    var_1 = var(fund_net_list, 1)
                    down_risk_ = down_std(fund_net_list)
                    down_std_ = ex_down_std(fund_net_list, benchmark_net_list)
                    alpha_beta = alpha_and_beta(fund_net_list, benchmark_net_list)
                    yearsharpRatio_ = yearsharpRatio(fund_net_list)
                    sortinoratio_ = sortinoratio(fund_net_list)
                    ir_ = ir(fund_net_list, benchmark_net_list)
                    rd_ = rd(fund_net_list, day_count)
                    treynor_ = treynor(fund_net_list, benchmark_net_list)

                    ret.append(idx)
                    ret.append(str(now)[:10])
                    ret.append(cumror_)
                    ret.append(ex_cumror_)
                    ret.append(annror_)
                    ret.append(ex_annror_)
                    ret.append(kurtosis_)
                    ret.append(ex_kurtosis_)
                    ret.append(skewness_)
                    ret.append(ex_skewness_)
                    ret.append(positive_ratio_)
                    ret.append(ex_positive_ratio_)
                    ret.append(volatility_)
                    ret.append(ex_volatility_)
                    ret.append(maxretrace_)
                    ret.append(avgretrace_)
                    ret.append(var_1)
                    ret.append(down_risk_)
                    ret.append(down_std_)
                    ret.append(alpha_beta[0])
                    ret.append(alpha_beta[1])
                    ret.append(yearsharpRatio_)
                    ret.append(sortinoratio_)
                    ret.append(ir_)
                    ret.append(rd_)
                    ret.append(treynor_)
                    ret.append(p)
                    res.append(ret)
                except:
                    ret.append(idx)
                    ret.append(str(now)[:10])
                    ret.append(np.nan)
                    ret.append(np.nan)
                    ret.append(np.nan)
                    ret.append(np.nan)
                    ret.append(np.nan)
                    ret.append(np.nan)
                    ret.append(np.nan)
                    ret.append(np.nan)
                    ret.append(np.nan)
                    ret.append(np.nan)
                    ret.append(np.nan)
                    ret.append(np.nan)
                    ret.append(np.nan)
                    ret.append(np.nan)
                    ret.append(np.nan)
                    ret.append(np.nan)
                    ret.append(np.nan)
                    ret.append(np.nan)
                    ret.append(np.nan)
                    ret.append(np.nan)
                    ret.append(np.nan)
                    ret.append(np.nan)
                    ret.append(np.nan)
                    ret.append(np.nan)
                    ret.append(p)
                    res.append(ret)

            else:
                ret.append(idx)
                ret.append(str(now)[:10])
                ret.append(np.nan)
                ret.append(np.nan)
                ret.append(np.nan)
                ret.append(np.nan)
                ret.append(np.nan)
                ret.append(np.nan)
                ret.append(np.nan)
                ret.append(np.nan)
                ret.append(np.nan)
                ret.append(np.nan)
                ret.append(np.nan)
                ret.append(np.nan)
                ret.append(np.nan)
                ret.append(np.nan)
                ret.append(np.nan)
                ret.append(np.nan)
                ret.append(np.nan)
                ret.append(np.nan)
                ret.append(np.nan)
                ret.append(np.nan)
                ret.append(np.nan)
                ret.append(np.nan)
                ret.append(np.nan)
                ret.append(np.nan)
                ret.append(p)
                res.append(ret)

    result = pd.DataFrame(res)
    result.columns = ['code', 'EndDate', 'cumror', 'ex_cumror'
        , 'annror', 'ex_annror', 'kurtosis', 'ex_kurtosis'
        , 'skewness', 'ex_skewness', 'positive_ratio'
        , 'ex_positive_ratio', 'volatitility'
        , 'ex_volatitility', 'maxretrace', 'avgretrace'
        , 'var_1', 'down_std'
        , 'ex_down_std', 'alpha', 'beta', 'yearsharp'
        , 'sortinoratio', 'ir', 'rd', 'treynor', 'period']

    fund_name = fund_id[['main_code', 'name', 'label', 'start_date']]
    fund_name = fund_name.rename(columns={'main_code': 'code'})
    # result = pd.merge(fund_name, result,on='code',how='inner')
    # result.to_csv('value_data.csv')
    # result = pd.read_csv('value_data.csv',encoding='gbk',index_col=0)
    # =============================================================================
    # topsis 模型 
    # ============================================================================
    INDICATORS = ['cumror', 'ex_cumror'
        , 'annror', 'ex_annror', 'kurtosis', 'ex_kurtosis'
        , 'skewness', 'ex_skewness', 'positive_ratio'
        , 'ex_positive_ratio', 'volatitility'
        , 'ex_volatitility', 'maxretrace', 'avgretrace'
        , 'var_1', 'down_std'
        , 'ex_down_std', 'alpha', 'beta', 'yearsharp'
        , 'sortinoratio', 'ir', 'rd', 'treynor']

    INDICATORS_AFTER_SELECT = ['ex_annror_36', 'alpha_36', 'ex_skewness_12',
                               'volatitility_12', 'var_1_6', 'down_std_6',
                               'avgretrace_60', 'ir_60', 'sortinoratio_6',
                               'rd_24']

    direc = [1, 1, 1, -1, -1, -1, -1, 1, 1, 1]
    w = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    col = ['code', 'day', 'M_O_Date']
    col.extend(INDICATORS_AFTER_SELECT)
    break_point_lst = [0.15, 0.70, 0.85, 0.95]  # 评级标准-分位点设置
    # 读取基金评价指标数据
    df = result.copy()
    # 对指标按照滚动窗口期进行切分
    df, indicators = data_period_split(df, INDICATORS)
    df['M_O_Date'] = df.day.apply(lambda s: s[:7])
    df = pd.merge(fund_name, df, on='code', how='inner')

    df_after_select = df.copy()
    label_lst = df_after_select.label.drop_duplicates().tolist()
    # 计算生成基金ＴＯＰＳＩＳ评分
    ret = list()
    for i in label_lst:
        print(i)
        tmp = df_after_select[df_after_select['label'] == i].copy()
        tmp = no_extreme(tmp, indicators)
        tmp.iloc[:, 5:-1] = tmp.iloc[:, 5:-1] \
            .apply(lambda x: (x - x.mean()) / x.std())

        # tmp = df_after_select.query("underlying_asset_type_id == %s"%i)
        funds_topsis_score = generate_topsis_score(tmp, col, INDICATORS_AFTER_SELECT, direc, w)
        # 基于评分给定基金评级
        funds_topsis_rank = rank_stars(funds_topsis_score, 'topsis', break_point_lst)
        ret.append(funds_topsis_rank)
    # 输出结果 
    fund_rank_r = pd.concat(ret)
    # fund_rank_r.code = fund_rank_r.code.apply(lambda s: str(s).zfill(6))
    fund_rank_r = pd.merge(fund_name, fund_rank_r, on='code', how='inner')
    # fund_rank_r = pd.merge(fund_id[['main_code','name','label']], fund_rank_r, left_on='main_code',right_on='code', how='inner')
    fund_rank_r.to_csv('result/fund_rank_by_topsis_v5_' + today +'.csv', encoding='gbk')
# =============================================================================
# 基金精选池
# =============================================================================
    # fund_rank_r = pd.read_csv('result/fund_rank_by_topsis_v5_' + today +'.csv', encoding='gbk',index_col=0)
    stars_select_1 = fund_rank_r.query("stars >= 4")

    # 计算基金抗风险评分-基于动态回撤
    ret_dydr = list()
    for f in stars_select_1.code:
        print(f)
        res = list()
        nv = fund_value_bench.query("code == '%s'"%f)
        nv.sort_values(by='day', inplace=True)
        if len(nv) > 60 * 20 * 0.8:
            tmp_nv = nv[['code', 'day', 'sum_value']].copy()
            tmp_nv['log'] = tmp_nv.sum_value.apply(lambda s: np.log(s))
            tmp_nv['log_return'] = tmp_nv['log'].diff()
            tmp_nv['log_return_abs'] = tmp_nv['log_return'].apply(lambda s: abs(s))
            if tmp_nv.log_return_abs.sum() > 0:
                dd = cal_dynamic_drawdown_ind(f, tmp_nv)
                res.append(f)
                res.append(dd[0])
                res.append(dd[1])
                res.append(dd[2])
                res.append(dd[3])
                ret_dydr.append(res)
    dynamic_draw_any = pd.DataFrame(ret_dydr, columns=['code', 'draw_days_max', 'draw_days_mean',
                                                      'draw_max', 'draw_mean'])
    # dynamic_draw_any = str_code(dynamic_draw_any)

    # 计算动态回撤率衍生评分
    
    info_draw_anly = fund_name.merge(dynamic_draw_any, on='code')
    ret_dy_topsis = list()
    for idx, group_ in info_draw_anly.groupby('label'):
        tmp = group_.copy()
        zb_lst = ['draw_days_max', 'draw_days_mean', 'draw_max', 'draw_mean']
        direc = [-1, -1, 1, 1]
        w = [1, 1, 1, 1]
        df = topsis(tmp, zb_lst, direc, w)
        ret_dy_topsis.append(df)
    fund_dy_analysis = pd.concat(ret_dy_topsis)
    fund_dy_analysis = fund_dy_analysis.rename(columns={'topsis': 'dy_score'})

# 第一阶段筛选
    ret_fisrt = list()
    for idx, group in fund_dy_analysis.groupby('label'):
        tmp = group.sort_values(by='dy_score', ascending=False).head(30)
        ret_fisrt.append(tmp)
    df_first = pd.concat(ret_fisrt)
# 第二阶段，进攻能力筛选
# 计算基金抗风险评分-进攻能力
    ret_aggressive = list()
    for f in df_first.code:
        print(f)
        res = list()
        nv = fund_value_bench.query("code == '%s'"%f)
        nv.sort_values(by='day', inplace=True)
        # 生成工作数据
        nv['M_O_Date'] = nv.day.apply(lambda s: str(s)[:7])
        df = nv.copy()
        df.iloc[:, 2:4] = df.iloc[:, 2:4].pct_change()
        df_ = df.dropna()
        if len(df_) > 60 * 20 * 0.8:
            rev_m = df_.groupby('M_O_Date')['sum_value', 'b_mark'] \
                .apply(lambda x: (1 + x).prod() - 1).reset_index()
            rev_m['ex_chg'] = rev_m['sum_value'] - rev_m['b_mark']
            positive_dis = max(rev_m.b_mark.quantile(0.6), 0)
            negative_dis = min(rev_m.b_mark.quantile(0.4), 0)

            # 定义进攻能力
            positive = rev_m.query("b_mark >= %s & b_mark <= %s" % (negative_dis, positive_dis))
            p_ratio = len(positive.query('ex_chg > 0')) / len(positive)
            p_median = positive.ex_chg.quantile(0.5)
            res.append(f)
            res.append(p_ratio)
            res.append(p_median)
            ret_aggressive.append(res)
    aggressive = pd.DataFrame(ret_aggressive, columns=['code', 'p_ratio', 'p_median'])

    # 计算进攻衍生评分
    info_ag_anly = fund_name.merge(aggressive, on='code')
    ret_ag_topsis = list()
    for idx, group_ in info_ag_anly.groupby('label'):
        tmp = group_.copy()
        zb_lst = ['p_ratio', 'p_median']
        direc = [1, 1]
        w = [1, 1]
        df = topsis(tmp, zb_lst, direc, w)
        ret_ag_topsis.append(df)
    fund_ag_analysis = pd.concat(ret_ag_topsis)
    fund_ag_analysis = fund_ag_analysis.rename(columns={'topsis': 'ag_score'})
    
    
    # 第二阶段筛选
    ret_second = list()
    for idx, group in fund_ag_analysis.groupby('label'):
        tmp = group.sort_values(by='ag_score', ascending=False).head(10)
        ret_second.append(tmp)
    df_second = pd.concat(ret_second)
    
    df_second.to_excel('result/good_funds_' + today + '.xls',encoding='gbk')
