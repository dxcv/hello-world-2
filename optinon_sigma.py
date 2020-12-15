#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 10:38:59 2019
计算认购和认沽隐含不波动率 并与上证50的波动率进行对比
@author: yeecall
"""

from __future__ import division
import pandas as pd
import os
import numpy as np
import datetime
import talib as tb
import matplotlib.pyplot as plt
from scipy.stats import norm
from jqdatasdk import *
auth('18610039264','zg19491001')

def cal_sig(t,s0,X,mktprice,kind):
    # 设定参数
    r=0.032 # risk-free interest rate
#    t=float(30)/365 # time to expire (30 days)
    q=0 # dividend yield
    S0=s0 # underlying price 正股价
#    X=2.2 # strike price 行权价
#    mktprice=0.18 # market price
    
    # 用二分法求implied volatility，暂只针对call option
    if kind==0:
        sigma=0.3 # initial volatility
        C=P=0 
        upper=1 
        lower=0 
        num=0
        while (abs(C-mktprice)>1e-3) & (num<50):
#            print(abs(C-mktprice))
            d1=(np.log(S0/X)+(r-q+sigma**2/2)*t)/(sigma*np.sqrt(t))
            d2=d1-sigma*np.sqrt(t)
            C=S0*np.exp(-q*t)*norm.cdf(d1)-X*np.exp(-r*t)*norm.cdf(d2)
#            P=X*np.exp(-r*t)*norm.cdf(-d2)-S0*np.exp(-q*t)*norm.cdf(-d1)
            if C-mktprice>0: 
                upper=sigma
                sigma=(sigma+lower)/2
            else:
                lower=sigma
                sigma=(sigma+upper)/2
            if sigma<1e-3:
                sigma=sigma+1e-1
            num=num+1
        return sigma # implied volatility
    elif kind==1:
        sigma=0.3 # initial volatility
        C=P=0 
        upper=1 
        lower=0 
        num=0
        while (abs(P-mktprice)>1e-3) & (num<50):
#            print(abs(P-mktprice))
            d1=(np.log(S0/X)+(r-q+sigma**2/2)*t)/(sigma*np.sqrt(t))
            d2=d1-sigma*np.sqrt(t)
#            C=S0*np.exp(-q*t)*norm.cdf(d1)-X*np.exp(-r*t)*norm.cdf(d2)
            P=X*np.exp(-r*t)*norm.cdf(-d2)-S0*np.exp(-q*t)*norm.cdf(-d1)
#            print(P)
            if P-mktprice>0: 
                upper=sigma
                sigma=(sigma+lower)/2
            else:
                lower=sigma
                sigma=(sigma+upper)/2
            if sigma<1e-3:
                sigma=sigma+1e-1
            num=num+1
#            print(sigma)
#            print('+++++')
        return sigma # implied volatility
    
    
#获取交易日  
def tradeday(sday,eday):
  """
  输入 开始时间 和 截止时间
  输出 list 交易日 datetime格式
  """
  return get_trade_days(sday,eday)

def stock_price(sec,period,sday,eday):
  """
  输入 股票代码，开始日期，截至日期
  输出 个股的后复权的开高低收价格
  """
  temp= get_price(sec, start_date=sday, end_date=eday, frequency=period, fields=['open', 'close', 'low', 'high', 'volume','money','high_limit'], skip_paused=False, fq='pre', count=None).reset_index()\
                     .rename(columns={'index':'tradedate'})
  temp['stockcode']=sec
  return temp

def cal_dis(row):
    CO=cal_sig(row['time_delt'],row['etf_close'],row['exercise_price'],row['close_CO'],0)
#    print('______')
    PO=cal_sig(row['time_delt'],row['etf_close'],row['exercise_price'],row['close_PO'],1)
    return ([CO,PO])


if __name__=='__main__':
    # 改成本地的路径
    pyth='/Users/yeecall/Documents/mywork/joinquant_data'
    os.chdir(pyth)
    
    sday='2018-01-01'
    eday=str(datetime.datetime.today())[:10]
    tradedays=tradeday(sday,eday)
    tradeday_lst=[i.strftime('%Y-%m-%d') for i in tradedays]
    period_vol=5
    
    q=query(opt.OPT_CONTRACT_INFO).filter(opt.OPT_CONTRACT_INFO.exercise_date>=sday
           ,opt.OPT_CONTRACT_INFO.underlying_type=='ETF')
    df1=opt.run_query(q)
#    print(df)
#    df.to_excel('option_contet.xls')
    df=df1[['code','name','contract_type','exchange_code','underlying_type','exercise_price'
           ,'list_date','expire_date']]
    df=df.query("exchange_code=='XSHG' & underlying_type=='ETF'")
    df.list_date=df.list_date.apply(lambda s:str(s))
    df.expire_date=df.expire_date.apply(lambda s:str(s))
   
    # 交易期权列表
    option_lst=df.code.tolist()
    for i in range(len(option_lst)):
        print(option_lst[i])
        q=query(opt.OPT_DAILY_PRICE.code,
        opt.OPT_DAILY_PRICE.date,
        opt.OPT_DAILY_PRICE.high,
        opt.OPT_DAILY_PRICE.open,
        opt.OPT_DAILY_PRICE.low,
        opt.OPT_DAILY_PRICE.close,
        opt.OPT_DAILY_PRICE.change_pct_close,
        opt.OPT_DAILY_PRICE.volume,
        opt.OPT_DAILY_PRICE.position).filter(opt.OPT_DAILY_PRICE.code==option_lst[i]).order_by(opt.OPT_DAILY_PRICE.date.desc())
        temp=opt.run_query(q)
        if i==0:
            op_price=temp.copy()
        else:
            op_price=pd.concat([op_price,temp.copy()])
    op_price.to_csv('data/op_price_day_{var1}_{var2}.csv'.format(var1=sday,var2=eday))       
    op_price=op_price.rename(columns={'date':'day'})     
    op_price.day=op_price.day.apply(lambda s:str(s)[:10])      
    # 获取50ETF的数据
    price_50ETF=stock_price('510050.XSHG','1d',sday,eday)
    price_50ETF.tradedate=price_50ETF.tradedate.apply(lambda s:str(s)[:10])
    price_50ETF_s=price_50ETF[['tradedate','close']]
    price_50ETF_s.columns=['day','etf_close']
    
    
    
    ret=list()
    for d in tradeday_lst:
        print(d)
        res=list()
        temp=df.query("list_date<='{var1}' & expire_date>='{var1}'".format(var1=d)).copy()
        temp_CO=temp[temp['contract_type']=='CO']
        temp_CO=temp_CO[['code','exercise_price','list_date','expire_date']]
        temp_CO.columns=['code_CO','exercise_price','list_date','expire_date']
        temp_PO=temp[temp['contract_type']=='PO']
        temp_PO=temp_PO[['code','exercise_price','list_date','expire_date']]
        temp_PO.columns=['code_PO','exercise_price','list_date','expire_date']

        data_heyue=temp_CO.merge(temp_PO,on=['exercise_price','list_date','expire_date'])
        data_heyue['day']=d
        data_heyue['time_delt']=pd.to_datetime(data_heyue['expire_date'])-pd.to_datetime(data_heyue['day'])
        data_heyue.time_delt=data_heyue.time_delt.apply(lambda s:s.days)
        data_heyue=data_heyue.merge(price_50ETF_s,on=['day'])
        op_price_1=op_price.copy()
        op_price_1=op_price_1.rename(columns={'code':'code_CO'})
        data_heyue1=data_heyue.merge(op_price_1,on=['code_CO','day'])
        data_heyue1=data_heyue1.rename(columns={'close':'close_CO','volume':'position_CO'})
        op_price_2=op_price.copy()
        op_price_2=op_price_2.rename(columns={'code':'code_PO'})
        data_heyue2=data_heyue1.merge(op_price_2,on=['code_PO','day'])
        data_heyue2=data_heyue2.rename(columns={'close':'close_PO','volume':'position_PO'})

        data_heyue2['time_delt']=data_heyue2.time_delt/365
#        vol_sum=data_heyue2.position_CO.sum()+data_heyue2.position_PO.sum()
        data_heyue2['position_CO']=data_heyue2.position_CO/data_heyue2.position_CO.sum()
        data_heyue2['position_PO']=data_heyue2.position_PO/data_heyue2.position_PO.sum()
        
        data_heyue2['op_var_dis']=[cal_dis(row) for idx,row in data_heyue2.iterrows()]
        
        data_heyue2['op_co']=[row['op_var_dis'][0] for idx,row in data_heyue2.iterrows()]
        data_heyue2['op_po']=[row['op_var_dis'][1] for idx,row in data_heyue2.iterrows()]
        
        
        data_heyue2['co_position']=data_heyue2['op_co']*data_heyue2['position_CO']
        data_heyue2['po_position']=data_heyue2['op_po']*data_heyue2['position_PO']
        
        res.append(d)
        res.append(data_heyue2.co_position.sum())
        res.append(data_heyue2.po_position.sum())
        ret.append(res)
    tTM=pd.DataFrame(ret)
    tTM.columns=['day','co_sigma','po_sigma']
    
    c=price_50ETF_s.set_index('day')
    c['logreturn']=np.log(c).etf_close-np.log(c).etf_close.shift(1)
    c['volatility']=tb.STDDEV(c['logreturn'].values,period_vol,nbdev=1)
    c['volatility']=c['volatility']/((1/252)**0.5)
    c=c.reset_index()
    out=tTM.merge(c.loc[:,['day','volatility']],on='day')
    
    out.day=out.day.apply(lambda s:datetime.datetime.strptime(s,'%Y-%m-%d'))

    
    out = out[['day','co_sigma','po_sigma','volatility']]


    out['day'] = pd.to_datetime(out['day'])
    out = out.set_index(['day'], drop=True)
#    out.plot()
    
    ma_out=out.assign(ma1=tb.MA(out['co_sigma'].values,5,matype=1))\
              .assign(ma2=tb.MA(out['co_sigma'].values,12,matype=1))
    ma_out=ma_out.query("co_sigma>0")
    ma_out.loc[:,['day','ma1','ma2']].plot()

#    out.columns=['日期','认购期权隐含波动率','认沽期权隐含波动率','历史波动率']
    ma_out.to_excel('option_ma.xls',encoding='gbk')
    
    
