# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 16:22:26 2020
优秀基金个股配置统计分析
@author: Administrator
"""
from __future__ import division
import pandas as pd
import numpy as np
from datetime import datetime
from jqdatasdk import *
# auth('18610039264','zg19491001')
auth('15168322665','Juzheng2018')

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
    return df

def fund_hold_HK(code, date):
    q = query(finance.FUND_PORTFOLIO_STOCK.code, 
              finance.FUND_PORTFOLIO_STOCK.period_end,
              finance.FUND_PORTFOLIO_STOCK.symbol,
              finance.FUND_PORTFOLIO_STOCK.name,
              finance.FUND_PORTFOLIO_STOCK.proportion).filter(finance.FUND_PORTFOLIO_STOCK.code==code,
                                                              finance.FUND_PORTFOLIO_STOCK.period_end==date)
    df = finance.run_query(q)    
    df.drop_duplicates(inplace=True)
    return df


def all_fund_hold_analysis(code_lst, pub_date):
    ret = list()
    for i in code_lst:
        print(i + ' is cal of holding')
        # print(fund_hold_(i, pub_day))
        ret.append(fund_hold_(i, pub_date))
    fund_hold = pd.concat(ret)
    fund_hold.symbol = fund_hold.symbol.apply(normalize_code)
    # 持仓个股分析
    # index_weight
    indust_info = get_industries(name='sw_l1', date=pub_date).reset_index().rename(columns={'index':'industry_code'}) 
    indust_info =indust_info[['industry_code', 'name']]
    indust_info.columns = ['indust', 'indust_name'] 
    
    fund_hold_ratio = fund_hold.groupby(['symbol', 'name'])['proportion'].sum().reset_index()

    fund_hold_counts = fund_hold.groupby(['symbol', 'name'])['proportion'].count().reset_index()
    fund_hold_counts = fund_hold_counts.rename(columns={'proportion':'counts'})
    fund_hold_stat = fund_hold_ratio.merge(fund_hold_counts,on=['symbol', 'name'])
    fund_hold_stat['proportion'] = fund_hold_stat['proportion'] / len(code_lst)
    # 获取参考标的权重
    index_weight = get_index_weights(index_id=benchmark_code, date=pub_date).reset_index()
    # fund_hold_stat.symbol = fund_hold_stat.symbol.apply(normalize_code)
    fund_hold_stat = fund_hold_stat.rename(columns={'symbol':'code'})
    stat = fund_hold_stat.merge(index_weight[['code','weight']], how='left')
    stat['diff'] = stat['proportion']  / stat['weight']   
    stat['indust'] = stat.code.apply(check_industry)
    stat = stat.merge(indust_info)
    stat = stat.sort_values(by='diff',ascending=False)
    # 持仓行业分析
    # fund_hold.symbol = fund_hold.symbol.apply(normalize_code)
    # fund_hold['indust'] = fund_hold.symbol.apply(check_industry)
    # tmp = fund_hold.groupby(['indust', 'code'])['proportion'].sum().reset_index()
    # tmp_1 = tmp.groupby('indust')['proportion'].mean().reset_index()
    # tmp_2 = tmp.groupby('indust')['proportion'].count().reset_index()
    # fund_hold_indust = tmp_1.merge(tmp_2, on='indust')
    # fund_hold_indust.columns=['indust', 'proportion', 'counts']


    
    # index_weight['indust'] = index_weight.code.apply(check_industry)
    # index_weight_industry = index_weight.groupby('indust')['weight'].sum().reset_index()
    # fund_hold_indust = fund_hold_indust.merge(index_weight_industry)  
    # fund_hold_indust['diff'] = fund_hold_indust['proportion'] * fund_hold_indust['counts'] / fund_hold_indust['weight']
    # ret = fund_hold_indust.merge(indust_info)
    # ret = ret.sort_values(by='diff',ascending=False) 
    return stat
    
def check_industry(code, indust_classify_code='sw_l1'):
    try:
        return get_industry(code)[code][indust_classify_code]['industry_code']
    except:
        return 0


if __name__ == '__main__':
    
    # 设置变量
    benchmark_code = '000300.XSHG'
    pub_date = '2020-09-30'
    # 数据加载区间
    sdate = '2010-01-01'
    edate = '2020-09-30'
    # 基金池列表
    fund_num = 7
    # 基金列表
    if fund_num == 1:
        code_lst = ['000410', '519732', '000390', '200016', '000751', '001508',
                    '213001', '260101']
        
    elif fund_num == 2:          
        code_lst = ['320012', '519732', '000362', '202023', '000751', '000577']
    elif fund_num == 3:
        code_lst = ['180012','163412','000595','110011','000410','166002',
                    '200016','519697','540006','000577','000751']
    elif fund_num == 4:
        tmp = pd.read_csv('good_funds.csv',encoding='gbk')
        tmp.code = tmp.code.apply(lambda s: str(s).zfill(6))
        code_lst = tmp.code.tolist()[:36]
    elif fund_num == 5:
        code_lst = ['519732', '202023', '070002', '519697', '000751', '001182',
                    '100016', '160133','001186','000513']
    elif fund_num == 6:
        tmp = pd.read_csv('fund_industry.csv',encoding='gbk')
        fund_stars = pd.read_csv('fund_rank_by_topsis_v5_2020-07-13.csv',encoding='gbk',index_col=0)
        tmp = tmp.merge(fund_stars[['main_code','stars']],left_on='InnerCode',right_on='main_code')
        tmp_ind = tmp.query("(proportion > 0.7) & (indust_name == '医药生物I')")
        tmp_ind = tmp_ind.query("stars > 3")
        tmp_ind.InnerCode = tmp_ind.InnerCode.apply(lambda s: str(s).zfill(6))
        code_lst = tmp_ind.InnerCode.tolist()
    elif fund_num == 7:
        # 港股
        code_lst = ['519779', '004476', '004340', '001875','004477','002685','001703','003413']
        ret = list()
        for i in code_lst:
            print(i + ' is cal of holding')
            # print(fund_hold_(i, pub_day))
            ret.append(fund_hold_HK(i, pub_date))
        fund_hold = pd.concat(ret)
        fund_hold_mean = fund_hold[['symbol','name','proportion']].groupby(['symbol','name']).sum()/len(code_lst)
        fund_hold_mean = fund_hold_mean.reset_index()
        fund_hold_mean['sign'] = fund_hold_mean.symbol.apply(lambda s:len(str(s)))
        fund_hold_mean = fund_hold_mean[fund_hold_mean['sign'] == 5]
        fund_hold_mean = fund_hold_mean.sort_values(by='proportion',ascending=False)
        fund_hold_count = fund_hold[['symbol','name','proportion']].groupby(['symbol','name']).count()
        fund_hold_count = fund_hold_count.reset_index()
        fund_hold_count['sign'] = fund_hold_count.symbol.apply(lambda s:len(str(s)))
        fund_hold_count = fund_hold_count[fund_hold_count['sign'] == 5]
        fund_hold_count = fund_hold_count.sort_values(by='proportion',ascending=False)
        fund_hold_count.to_excel(pub_date + '_' + str(fund_num) + '_' + 'good_fund_hold_analysis'+ ".xls",encoding='gbk')
    if fund_num != 7:
        st = all_fund_hold_analysis(code_lst, pub_date)
        st.to_excel(pub_date + '_' + str(fund_num) + '_' + 'good_fund_hold_analysis'+ ".xls",encoding='gbk')

