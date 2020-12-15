# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 16:36:15 2020
对优秀的基金历史持仓进行回测，做新的牛基股策略
@author: Administrator
"""
from __future__ import division
import pandas as pd
import numpy as np
import datetime
from jqdatasdk import *
auth('18610039264', 'zg19491001')

def history_pub_day(s_year,e_year):
    pub_day_lst = list()
    for y in range(s_year,e_year):
        for m in [3, 6, 9, 12]:
            if m in [3,12]:  
                pub_day_lst.append(str(y) + '-' + str(m).zfill(2) + '-' + str(31))
            else:
                pub_day_lst.append(str(y) + '-' + str(m).zfill(2) + '-' + str(30))
    return pub_day_lst           

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

def check_industry(code, indust_classify_code='sw_l1'):
    try:
        return get_industry(code)[code][indust_classify_code]['industry_code']
    except:
        return 0

# 获取当前全部股票信息
def all_stocks(today):
    """
    输出股票代码，股票名称，股票简称，上市时间等主要字段
    输出格式 dataframe
    """
    return get_all_securities(types=[], date=today).reset_index().rename(columns={'index': 'stockcode'})
    
def next_pub_day_def(pub_day):
    year = int(pub_day[:4])
    month = int(pub_day[5:7])
    if month == 12:
        year = year + 1
        month = 3
        day = 31
    else:
        month = month + 3
        if month in [6, 9]:
            day = 30
        else:
            day = 31
    rt = str(year) + '-' + str(month).zfill(2) + '-' + str(day).zfill(2)
    return rt

def month_add_one(date):
    year = int(date[:4])
    month = int(date[5:7])
    if month == 12:
        year = year + 1
        month = 1
        day = 31
    elif month == 3:
        month = 4
        day = 30
    elif month == 6:
        month = 7
        day = 31
    elif month == 9:
        month = 10
        day = 31
    rt = str(year) + '-' + str(month).zfill(2) + '-' + str(day).zfill(2)
    return rt
    

def stock_price(sec, period, sday, eday):
    temp = get_price(sec, start_date=sday, end_date=eday, frequency=period, fields=['open', 'close', 'high', 'low'],
                     skip_paused=True, fq='pre', count=None).reset_index() \
        .rename(columns={'index': 'tradedate'})
    temp['stockcode'] = sec
    return temp

def backtest(df, holds):
    test = df[['period_end','symbol']]
    test.period_end = test.period_end.apply(lambda s: str(s)[:10])
    test = test.sort_values(by='period_end')
    # test['e_ddy'] = test.period_end.apply(lambda s : next_pub_day_def(s))
    
    buy_time_lst = test.period_end.drop_duplicates().tolist()
    out = list() # 组合收益率
    for bt in buy_time_lst:
        print(bt)
        st = next_pub_day_def(bt)
        stock_chg = list()
        tmp = test[test['period_end'] == bt]
        stocks = tmp.symbol.tolist()
        stocks = [ normalize_code(s) for s in stocks]
        for s in stocks:
            stk_price = stock_price(s, '1d', bt, st)
            stk_price = stk_price[['tradedate','close','stockcode']]
            stock_chg.append(stk_price)
        stock_df = pd.concat(stock_chg)
        stock_df_pivot = pd.pivot_table(stock_df, index='tradedate', columns='stockcode',values='close')
        stock_df_pivot = stock_df_pivot.diff() / stock_df_pivot.shift(1)
        stock_df_pivot.fillna(0,inplace=True)
        stock_df_pivot = stock_df_pivot * (1/holds)
        s_portpolio = stock_df_pivot.sum(axis=1).reset_index()
        s_portpolio.columns=['tradedate','chg']
        out.append(s_portpolio.iloc[1:])
    portpolio = pd.concat(out)
    portpolio['nv'] = (1 + portpolio['chg']).cumprod()
    portpolio = portpolio.set_index('tradedate')  
    portpolio['nv'].plot()
    return portpolio


def backtest_1(df):
    """
    按照季报日期 延后一个月，保证开仓股票与实际交易一致
    """
    test = df[['period_end','symbol']]
    test.period_end = test.period_end.apply(lambda s: str(s)[:10])
    test = test.sort_values(by='period_end')
    # test['e_ddy'] = test.period_end.apply(lambda s : next_pub_day_def(s))
    
    buy_time_lst = test.period_end.drop_duplicates().tolist()
    out = list() # 组合收益率
    for bt in buy_time_lst:
        print(bt)
        st = next_pub_day_def(bt)
        stock_chg = list()
        tmp = test[test['period_end'] == bt]
        bt = month_add_one(bt)
        st = month_add_one(st)
        holds = len(tmp)
        holds = max(15,holds)
        stocks = tmp.symbol.tolist()
        stocks = [ normalize_code(s) for s in stocks]
        for s in stocks:
            stk_price = stock_price(s, '1d', bt, st)
            stk_price = stk_price[['tradedate','close','stockcode']]
            stock_chg.append(stk_price)
        stock_df = pd.concat(stock_chg)
        stock_df_pivot = pd.pivot_table(stock_df, index='tradedate', columns='stockcode',values='close')
        stock_df_pivot = stock_df_pivot.diff() / stock_df_pivot.shift(1)
        stock_df_pivot.fillna(0,inplace=True)
        stock_df_pivot = stock_df_pivot * (1/holds)
        s_portpolio = stock_df_pivot.sum(axis=1).reset_index()
        s_portpolio.columns=['tradedate','chg']
        out.append(s_portpolio.iloc[1:])
    portpolio = pd.concat(out)
    portpolio['nv'] = (1 + portpolio['chg']).cumprod()
    portpolio = portpolio.set_index('tradedate')  
    portpolio['nv'].plot()
    return portpolio    

if __name__ == '__main__':
    # 时间
    today = str(datetime.datetime.today())[:10]
    
    # 全部股票信息
    alstock = all_stocks(today)
    alstock = alstock.iloc[:,:2]
    alstock.columns = ['symbol','name']
    alstock.symbol = alstock.symbol.apply(lambda s: s[:6])
    # 基金池
    switch = 2
    if  switch == 1:
        InnerCode_lst = ['000410', '519732', '000390', '200016', '000751', '001508',
                        '213001', '260101']
    elif switch == 2:
        InnerCode_lst = ['519732', '202023', '070002', '519697', '000751', '001182',
                         '100016', '160133','001186','000513']
    elif switch == 3:
        # 医药回测
        tmp = pd.read_csv('fund_industry.csv',encoding='gbk')
        fund_stars = pd.read_csv('fund_rank_by_topsis_v5_2020-07-13.csv',encoding='gbk',index_col=0)
        tmp = tmp.merge(fund_stars[['main_code','stars']],left_on='InnerCode',right_on='main_code')
        tmp_ind = tmp.query("(proportion > 0.7) & (indust_name == '医药生物I')")
        tmp_ind = tmp_ind.query("stars > 4")
        tmp_ind.InnerCode = tmp_ind.InnerCode.apply(lambda s: str(s).zfill(6))
        InnerCode_lst = tmp_ind.InnerCode.tolist()
    # 输出历史的季报时间
    pubday_lst = history_pub_day(2015,2021)
    pubday_lst = [i for i in pubday_lst if i < today]
    
    # 历史优选基金持仓
    ret_hold = list()
    for p in pubday_lst:
        for fd in InnerCode_lst:
            print(str(p) + '_' + str(fd) + ' is get holds')
            tmp = fund_hold_(fd, p)
            ret_hold.append(tmp)
    df_hold = pd.concat(ret_hold)
    
    # 统计平均持仓
    df_mean_holds = df_hold.groupby(['period_end','symbol'])['proportion'].sum().reset_index()
    df_funds_num = df_hold.drop_duplicates(subset=['period_end','code'])
    df_funds_num = df_funds_num.groupby('period_end')['code'].count().reset_index().rename(columns={'code':'funds'}) 
    df_mean_holds_1 = df_mean_holds.merge(df_funds_num)   
    df_mean_holds_1['proportion'] = df_mean_holds_1['proportion'] / df_mean_holds_1['funds']
    df_mean_holds_1['Industry'] = df_mean_holds_1.symbol.apply(normalize_code)
    df_mean_holds_1['Industry'] = df_mean_holds_1.Industry.apply(check_industry)
    # 统计持有个股个数
    df_counts_holds = df_hold.groupby(['period_end','symbol'])['proportion'].count().reset_index()
    df_counts_holds['Industry'] = df_counts_holds.symbol.apply(normalize_code)
    df_counts_holds['Industry'] = df_counts_holds.Industry.apply(check_industry)
# =============================================================================
# 输出交易信号
# =============================================================================
    # 取平均持仓比率最大的10个股票，不分行业
    # df_mean_holds_1 = df_mean_holds_1.merge(alstock)                                                                   
    # hold_mean = list()
    # for idx, group in df_mean_holds_1.groupby('period_end'):
    #     tmp = group.sort_values(by='proportion',ascending=False)
    #     hold_mean.append(tmp.head(15))
    # hold_mean_df = pd.concat(hold_mean)
    
    # 取平均持仓比率最大的10个股票，不分行业,存在筛选条件
    # df_mean_holds_1 = df_mean_holds_1.merge(alstock)                                                                   
    # hold_mean = list()
    # for idx, group in df_mean_holds_1.groupby('period_end'):
    #     group = group.query("proportion > 1")
    #     tmp = group.sort_values(by='proportion',ascending=False)
    #     hold_mean.append(tmp.head(10))
    # hold_mean_df = pd.concat(hold_mean)
    
    # 取平均持仓比率最小的10个股票，不分行业
    df_mean_holds_1 = df_mean_holds_1.merge(alstock)                                                                   
    hold_mean = list()
    for idx, group in df_mean_holds_1.groupby('period_end'):
        # group = group.query("proportion < 1")
        tmp = group.sort_values(by='proportion',ascending=False)
        hold_mean.append(tmp.tail(15))
    hold_mean_df = pd.concat(hold_mean)  
    
    # 取平均持仓比率最大的1个股票，分行业，不做筛选
    # df_mean_holds_1 = df_mean_holds_1.merge(alstock)                                                                   
    # hold_mean = list()
    # for idx, group in df_mean_holds_1.groupby(['period_end','Industry']):
    #     tmp = group.sort_values(by='proportion',ascending=False)
    #     hold_mean.append(tmp.head(1))
    # hold_mean_df = pd.concat(hold_mean) 
    
    # 取平均持仓比率最大的1个股票，分行业，筛选 proportion>1
    # df_mean_holds_1 = df_mean_holds_1.merge(alstock)                                                                   
    # hold_mean = list()
    # for idx, group in df_mean_holds_1.groupby(['period_end','Industry']):
    #     group = group.query("proportion > 0.7")
    #     tmp = group.sort_values(by='proportion',ascending=False)
    #     hold_mean.append(tmp.head(1))
    # hold_mean_df = pd.concat(hold_mean) 
    
    # 取持仓家数最多的10只股票，不分行业
    # df_counts_holds = df_counts_holds.merge(alstock)                                                                   
    # hold_counts = list()
    # for idx, group in df_counts_holds.groupby('period_end'):
    #     tmp = group.sort_values(by='proportion',ascending=False)
    #     tmp = tmp.query("proportion >=2")
    #     hold_counts.append(tmp.head(10))
    # hold_counts_df = pd.concat(hold_counts)
    
    # 取平均持仓比率最小的1个股票，分行业，不做筛选
    # df_mean_holds_1 = df_mean_holds_1.merge(alstock)                                                                   
    # hold_mean = list()
    # for idx, group in df_mean_holds_1.groupby(['period_end','Industry']):
    #     tmp = group.sort_values(by='proportion',ascending=False)
    #     hold_mean.append(tmp.tail(1))
    # hold_mean_df = pd.concat(hold_mean) 
# =============================================================================
# 回测效果
# =============================================================================
    # dd = backtest_1(hold_counts_df)
    ss = backtest_1(hold_mean_df)
  

        
    