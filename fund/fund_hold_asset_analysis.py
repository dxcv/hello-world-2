# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 13:23:23 2020
给定一个基金池，分析其资产，行业以及个股持仓统计
@author: Administrator
"""

import pandas as pd
import numpy as np
import datetime
from jqdatasdk import *
auth('18610039264', 'zg19491001')


def check_industry(code, indust_classify_code='sw_l1'):
    try:
        return get_industry(code)[code][indust_classify_code]['industry_code']
    except:
        return 0



def last_pub_day_def(pub_day):
    year = int(pub_day[:4])
    month = int(pub_day[5:7])
    if month == 3:
        year = year - 1
        month = 12
        day = 31
    else:
        month = month - 3
        if month in [6, 9]:
            day = 30
        else:
            day = 31
    rt = str(year) + '-' + str(month).zfill(2) + '-' + str(day).zfill(2)
    return rt


# def get_stock_ratio(code, pub_day):
#     this_season_stock = get_stocks_lfp(code, pub_day)
#     if this_season_stock['RatioInNV'].sum() > 0:
#         this_season_stock['RatioInNV'] = this_season_stock['RatioInNV'] / this_season_stock['RatioInNV'].sum()
#         return this_season_stock
#     else:
#         return []

def get_fund_portfolio(fund):
    df = finance.run_query(query(finance.FUND_PORTFOLIO.code,
                                 finance.FUND_PORTFOLIO.period_end,
                                 finance.FUND_PORTFOLIO.stock_rate).filter(finance.FUND_PORTFOLIO.code==fund))
    return df

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
    df = df.query("stock_market > 5")
    df = df.sort_values(by='proportion').tail(10)
    return df

def get_industry_ratio(code, pub_day):
    stock_tmp = fund_hold_(code, pub_day)
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

# 获取当前全部股票信息
def all_stocks(today):
    """
    输出股票代码，股票名称，股票简称，上市时间等主要字段
    输出格式 dataframe
    """
    return get_all_securities(types=[], date=today).reset_index().rename(columns={'index': 'stockcode'})

if __name__ == '__main__':
    # 计算基金资产配置信息
    # 入参
    switch = 2
    if  switch == 1:
        # 自己基于评级和看图选择的
        InnerCode_lst = ['000410', '519732', '000390', '200016', '000751', '001508',
                        '213001', '260101']
    elif switch == 2:
        # 基于基金精选池，并看收益归因，业绩来源于多个行业，优选股能力
        InnerCode_lst = ['519732', '202023', '070002', '519697', '000751', '001182',
                         '100016', '160133','001186','000513']
        
    else:
        tmp = pd.read_excel('result/good_funds_2020-07-13.xls',encoding='gbk',index_col=0)
        tmp = tmp[tmp.label.isin(['股票型','标准混合型','偏股混合型'])]
        InnerCode_lst = tmp.code.drop_duplicates().tolist()
        InnerCode_lst = [str(i).zfill(6) for i in InnerCode_lst]
    # 报告期
    pub_day = '2020-06-30'
    # 配置
    # 行业信息
    indust_info = get_industries(name='sw_l1', date=None).reset_index().rename(columns={'index':'industry_code'})
    indust_info =indust_info[['industry_code', 'name']]
    indust_info.columns = ['Industry', 'indust_name'] 
    
    # 全部股票信息
    alstock = all_stocks(pub_day)
    alstock = alstock.iloc[:,:2]
    alstock.columns = ['symbol','name']
    alstock.symbol = alstock.symbol.apply(lambda s: s[:6])
    # 衍生变量-计算上一个季报日
    last_pub_day = last_pub_day_def(pub_day)
    # 大类资产统计-股票资产
    stocks_ratio = list()
    # 大类资产统计-行业信息
    industry = list()
    industry_ratio = list()
    # 大类资产统计-个股
    stocks_detail_now = list()
    stocks_detail_last = list()
    stocks_detail_chg = list()

    for code in InnerCode_lst:
        print(code)       
        stocks_ret = list()
        tmp = get_fund_portfolio(code)
        # 输出股票资产统计
        if len(tmp) > 0:
            ratio_lst = tmp['stock_rate'].tolist()
            ratio_now = ratio_lst[-1]  # 当前股票比例
            ratio_chg = ratio_lst[-1] - ratio_lst[-2]  # 最近一季度股票持仓比例变化量
            ratio_quant = len([i for i in ratio_lst if i < ratio_now]) / len(ratio_lst)  # 当前股票占比的分位点
            stocks_ret.append(code)
            stocks_ret.append(pub_day)
            stocks_ret.append(ratio_now)
            stocks_ret.append(ratio_chg)
            stocks_ret.append(ratio_quant)
            stocks_ratio.append(stocks_ret)
        # 行业占比
        this_season = get_industry_ratio(code, pub_day)
        last_season = get_industry_ratio(code, last_pub_day)
        if (len(this_season) * len(last_season)) > 0:
            ss = this_season.merge(last_season, on=['InnerCode', 'Industry'])
            indust_chg = ss[['InnerCode', 'Industry', 'Date_x', 'proportion_x', 'proportion_y']]
            indust_chg['chg'] = indust_chg['proportion_x'] - indust_chg['proportion_y']
            ratio_chg_quant = indust_chg[['InnerCode', 'Date_x', 'Industry', 'chg']].rename(columns={'Date_x': 'Date'})
            industry_ratio.append(ratio_chg_quant)
            industry.append(this_season)
        #
        # 个股
        this_season_stock = fund_hold_(code, pub_day)
        last_season_stock = fund_hold_(code, last_pub_day)
        if (len(this_season_stock) * len(last_season_stock)) > 0:
            this_season_stock = this_season_stock[['code', 'period_end', 'symbol', 'proportion']].rename(
                columns={'code': 'InnerCode', 'period_end': 'Date'})
            last_season_stock = last_season_stock[['code', 'period_end', 'symbol', 'proportion']].rename(
                columns={'code': 'InnerCode', 'period_end': 'Date'})
            tt = this_season_stock.merge(last_season_stock, on=['InnerCode', 'symbol'])
            stock_chg = tt[['InnerCode', 'symbol', 'Date_x', 'proportion_x', 'proportion_y']]
            stock_chg['chg'] = stock_chg['proportion_x'] - stock_chg['proportion_y']
            stock_ratio_chg_quant = stock_chg[['InnerCode', 'symbol', 'Date_x', 'chg']].rename(
                columns={'Date_x': 'Date'})
            stocks_detail_now.append(this_season_stock)
            stocks_detail_last.append(last_season_stock)
            stocks_detail_chg.append(stock_ratio_chg_quant)

    # 全部基金股票持仓各统计
    stocks = pd.DataFrame(stocks_ratio, columns=['InnerCode', 'Date', 'ratio_now', 'ratio_chg', 'ratio_quant'])
    # 行业基金持仓各统计
    industries = pd.concat(industry)
    industries_chg = pd.concat(industry_ratio)
    # 个股持仓统计
    stocks_details_now = pd.concat(stocks_detail_now)
    stocks_details_last = pd.concat(stocks_detail_last)
    stocks_details_chg = pd.concat(stocks_detail_chg)

    # 输出统计数值
    # 当前平均仓位,仓位历史分位点平均及近一个季度仓位变化平均
    stocks_out = stocks[['ratio_now','ratio_chg','ratio_quant']].mean().reset_index()
    stocks_out.columns = ['col','data']
    # 行业平均持仓 以及 行业平均加减仓
    industries_len = len(industries.InnerCode.drop_duplicates())
    industries_sum = industries.groupby('Industry').proportion.sum().reset_index()
    industries_sum['proportion'] = industries_sum['proportion'] / industries_len
    industries_sum = industries_sum.merge(indust_info).sort_values(by='proportion',ascending=False)

    industries_chg_len = len(industries_chg.InnerCode.drop_duplicates())
    industries_chg_sum = industries_chg.groupby('Industry').chg.sum().reset_index()
    industries_chg_sum['chg'] = industries_chg_sum['chg'] / industries_chg_len
    industries_chg_sum = industries_chg_sum.merge(indust_info).sort_values(by='chg',ascending=False)

    # 个股平均持仓比例，平均加仓比例，以及个股被持有的基金个数，基金个数增减
    stocks_len = len(stocks_details_now.InnerCode.drop_duplicates())
    stocks_sum_now = stocks_details_now.groupby('symbol').proportion.sum().reset_index()
    stocks_sum_now['proportion'] = stocks_sum_now['proportion'] / stocks_len
    stocks_sum_now = stocks_sum_now.merge(alstock).sort_values(by='proportion',ascending=False)


    stocks_details_chg_len = len(stocks_details_chg.InnerCode.drop_duplicates())
    stocks_details_chg_sum = stocks_details_chg.groupby('symbol').chg.sum().reset_index()
    stocks_details_chg_sum['chg'] = stocks_details_chg_sum['chg'] / stocks_details_chg_len
    stocks_details_chg_sum = stocks_details_chg_sum.merge(alstock).sort_values(by='chg',ascending=False)

    stocks_details_now_count = stocks_details_now.groupby('symbol').proportion.count().reset_index().rename(
        columns={'proportion': 'counts'})
    stocks_details_now_count_1 = stocks_details_now_count.merge(alstock).sort_values(by='counts',ascending=False)

    stocks_details_last_count = stocks_details_last.groupby('symbol').proportion.count().reset_index().rename(
        columns={'proportion': 'counts'})

    stock_count_chg = stocks_details_now_count.merge(stocks_details_last_count, on='symbol', how='outer').fillna(0)
    stock_count_chg['chg'] = stock_count_chg['counts_x'] - stock_count_chg['counts_y']
    stock_count_chg = stock_count_chg.merge(alstock).sort_values(by='chg',ascending=False)
    