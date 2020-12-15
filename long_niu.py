# -*- coding: utf-8 -*-
"""
Created on Fri Feb 09 14:55:22 2018

@author: Administrator
"""

from __future__ import division
import pandas as pd
import os
import numpy as np
import statsmodels.api as sm
import math
import datetime
import talib
import matplotlib.pyplot as plt
from jqdatasdk import *
auth('18610039264','zg19491001')

# =============================================================================
# 计算市场上最安全的个股
# =============================================================================
def back_to_test(series):
    list1=list(series)
#    print list1
    MDD=0#最大回撤
    MAX=list1[0]
    for i in range(len(list1)):
        if list1[i]>MAX:
            MAX=list1[i]
        MDD1=(MAX-list1[i])/MAX
        if MDD1>MDD:
            MDD=MDD1
    MDD=round(MDD*100,2)
    return MDD


def back_to_test1(series):
    list1=series.tolist()
#    print list1
    MDDDAYS=[0]#最大回撤天数
    days=0
    MAX=list1[0]
    for i in range(len(list1)):
        if list1[i]>MAX:
            MAX=list1[i]
            MDDDAYS.append(days)
            days=0            
        else:
          days=days+1
    MDDDAYS.append(days)
    mddday=max(MDDDAYS)
    return mddday

def r2(y):
  x=pd.Series(range(len(y)))
  X=sm.add_constant(x)
  result = (sm.OLS(y,X)).fit()
  return [result.rsquared_adj,result.params[0]]
  
def stock_price_year(sec,sday,eday):
  """
  输入 股票代码，开始日期，截至日期
  输出 个股的后复权的开高低收价格
  """
  temp= get_price(sec, start_date=sday, end_date=eday, frequency='250d', fields=None, skip_paused=False, fq='post', count=None).reset_index()\
                     .rename(columns={'index':'tradedate'})
  temp['stockcode']=sec
  return temp

def stock_price_day(sec,sday,eday):
  """
  输入 股票代码，开始日期，截至日期
  输出 个股的后复权的开高低收价格
  """
  temp= get_price(sec, start_date=sday, end_date=eday, frequency='1d', fields=None, skip_paused=False, fq='post', count=None).reset_index()\
                     .rename(columns={'index':'tradedate'})
  temp['stockcode']=sec
  return temp




# 获取当前全部股票信息
def  all_stocks():
  """
  输出股票代码，股票名称，股票简称，上市时间等主要字段
  输出格式 dataframe
  """
  return get_all_securities(types=[], date=None).reset_index().rename(columns={'index':'stockcode'})


# 输出行业成分股
def industries_stocks(code):
  """
  输入 行业代码
  输出 list 个股代码
  """
  return get_industry_stocks(code)
# =============================================================================
# 取数
# =============================================================================
if __name__=='__main__':    
    start='2010-01-01'
    end=str(datetime.datetime.today())[:10]
     
#    jrj_pg936_conf = {
#        'server': '172.16.198.11',
#        'user': 'JRJ_PG',
#        'password': '47sTXSLIQiXPG9pY2SYh',
#        'database': 'PGenius',
#        'charset': 'cp936',
#    #    'charset': 'UTF8',
#        'as_dict': False,
#    }
#    
#    
#    
#    conn = pymssql.connect(**jrj_pg936_conf) 
#    sql="""select STOCKCODE,STOCKSNAME,ENDDATE,TRADE_DAYS,FAC_TCLOSE,FAC_TOPEN,FAC_THIGH,FAC_TLOW
#    from ANA_STK_MKT_YEAR
#    where ENDDATE>'2009-01-01'"""
#    
#    df=pd.read_sql(sql,conn)

    all_stocks=all_stocks()
    all_stock_codes=all_stocks.stockcode.tolist()
    df=stock_price_year(all_stock_codes[0],'2010-01-01',end)
    for i in all_stock_codes[1:]:
        print(i)
        temp=stock_price_year(i,'2010-01-01',end)
        df=pd.concat([df,temp])
    df=df.dropna()
    res=df.assign(years=0) \
          .assign(ups=lambda df:df.close>df.open) \
          .assign(ups=lambda df:df.ups.apply(lambda s:1 if s==True else 0))
    
    
    
    A=res.loc[:,['stockcode','years']].groupby('stockcode').count().reset_index()
    B=res.loc[:,['stockcode','ups']].groupby('stockcode').sum().reset_index()
    result=pd.merge(A,B,on='stockcode')
    result['up_ratio']=result['ups']/result['years']
    
    
    # 筛选走势好的个股
    #result.query('years>5').sort_values(by='up_ratio',ascending=False)
    
    #result['up_ratio'].hist()
    
    # =============================================================================
    # 统计回撤
    # =============================================================================
    df_stocks=stock_price_day(all_stock_codes[0],start,end)
    for i in all_stock_codes[1:]:
        print(i)
        temp=stock_price_day(i,start,end)
        df_stocks=pd.concat([df_stocks,temp])
    
    df_stocks=df_stocks.dropna()
    df_stocks.to_csv('stocks_data.csv')

    
    #df_stocks=df_stocks.query("ENDDATE>'2014-01-01'")
    df_stocks.close=df_stocks.close.apply(lambda s:math.log(s))  
    
    df_stocks_1=df_stocks.loc[:,['stockcode','close']].groupby('stockcode')\
                         .close \
                         .apply(back_to_test) \
                         .reset_index() 
    
    df_stocks_2=df_stocks.loc[:,['stockcode','close']].groupby('stockcode')\
                         .close \
                         .apply(back_to_test1) \
                         .reset_index() 
                         
    #df_stocks_3=df_stocks.loc[:,['STOCKCODE','FAC_TCLOSE']].groupby('STOCKCODE')\
    #                     .FAC_TCLOSE \
    #                     .apply(r2) \
    #                     .reset_index()                      
    
    s=[]
    k=[]
    code=[]
    for name,group in df_stocks.groupby('stockcode'):
      t=group.copy()
      t.index=range(len(t))
      s.append(r2(t['close'])[0])
      k.append(r2(t['close'])[1])
      code.append(name)
    r=[s,k,code]
    r=pd.DataFrame(r).T
    r.columns=['R','K','stockcode']
    
    
    df_stocks_1.columns=['stockcode','MDD']
    df_stocks_2.columns=['stockcode','MDDAY']
    result=pd.merge(result,df_stocks_1)
    result=pd.merge(result,df_stocks_2)
    result=pd.merge(result,r)
        
    total=len(result)
    temp=result.copy()
    temp=temp.assign(years_rank=100*(total-temp.years.rank(ascending=False)+1)*1/total) \
                 .assign(up_ratio_rank=100*(total-temp.up_ratio.rank(ascending=False)+1)*2/total) \
                 .assign(MDD_rank=100*(total-temp.MDD.rank(ascending=True)+1)/total) \
                 .assign(MDDAY_rank=100*(total-temp.MDDAY.rank(ascending=True)+1)*1/total) \
                 .assign(R_rank=100*(total-temp.R.rank(ascending=False)+1)*2/total) \
                 .assign(K_rank=100*(total-temp.K.rank(ascending=False)+1)*2/total) \
                 .assign(rank=lambda df:df.years_rank+df.up_ratio_rank+df.MDD_rank+df.MDDAY_rank+df.R_rank+df.K_rank)
    
    
    temp['long_niu_score']=temp['rank']/9
    temp.sort_values(by='rank',ascending=False)
    temp['long_niu_score'].hist(bins=50)
    
    
    #temp[['STOCKCODE','long_niu_score']].query('long_niu_score>80').sort_values(by='long_niu_score',ascending=False).to_excel('long_niu.xls')
    #
    #df_stocks.query("STOCKCODE=='300735'")
    
    temp.to_excel('result/long_niu.xls')
    # =============================================================================
    # 寻找各行业的龙头
    # =============================================================================
    indus_df=get_industries(name='sw_l1').reset_index()
    indus_df.columns=['indu_code','indu_name','start_date']
    indust_lst=indus_df.indu_code.tolist()

    indust_stocks=pd.DataFrame(industries_stocks(indust_lst[0]))
    indust_stocks['indust']=indust_lst[0]
    indust_stocks.columns=['stockcode','indust']
    for i in indust_lst[1:]:
        print(i)
        try:
            indust_stocks_temp=pd.DataFrame(industries_stocks(i))
            indust_stocks_temp['indust']=i
            indust_stocks_temp.columns=['stockcode','indust']
            indust_stocks=pd.concat([indust_stocks,indust_stocks_temp])
        except:
            print(i)
    indust_stocks.index=range(len(indust_stocks))
    print(indust_stocks)
    
    temp=temp.merge(indust_stocks)
    temp=temp.drop_duplicates(subset=['stockcode'])

    # =============================================================================
    # 提取行业龙头
    # =============================================================================
    res=temp.copy()
    res=res.query("long_niu_score>80 & years>2")
    out=pd.Series([])
    num=0
    res=res.loc[:,['stockcode','long_niu_score','indust']]
    for  name,group in res.groupby('indust'):
      if num==0:
        group=group.drop_duplicates(subset=['stockcode'])
        out=group.sort_values(by='long_niu_score',ascending=False).head(10).copy()
      else:
        group=group.drop_duplicates(subset=['stockcode'])
        out=pd.concat([out,group.sort_values(by='long_niu_score',ascending=False).head(10).copy()])
      num=num+1   
    #out=out.T
    indus_df.columns=['indust','indu_name','start_date']
    out=pd.merge(out,indus_df,on='indust')
    out=pd.merge(out,all_stocks.loc[:,['stockcode','display_name']],on='stockcode')
    
    out=out.query("long_niu_score>80")
    out=out[['stockcode','display_name','indu_name']]
    out.columns=[u'股票代码',u'股票名称',u'行业名称']
    out=out.drop_duplicates(subset=[u'股票代码'])
    out.to_excel('result/long_niu_industry.xls')
    
    
    
    
    