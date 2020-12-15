#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 16:26:13 2019
统计每日港股通每只个股持股变化
@author: yeecall
"""

from __future__ import division
import pandas as pd 
import os
import numpy as np
from jqdatasdk import *
auth('18610039264','zg19491001')
import datetime
import talib as tb



if __name__ =='__main__':

# =============================================================================
# 每日前十大港股通交易股票 资金连续流入情况统计
# =============================================================================
    today=datetime.datetime.today()
    end=str(today)[:10]
    count_=20 #寻找近几日的数据
    ma_N=5  # 连续几日港股通资金净买入（10大个股）
    ma_M=10 # 连续几日港股通资金买入（全样本）
    
    trade_lst=get_trade_days(end_date=end, count=count_)
    tst=[str(i)[:10] for i in trade_lst]
    
    q=query(finance.STK_EL_TOP_ACTIVATE).filter(finance.STK_EL_TOP_ACTIVATE.day==tst[0],
           finance.STK_EL_TOP_ACTIVATE.link_id <= 310002)
    df=finance.run_query(q)
    for i in tst[1:]:
        q=query(finance.STK_EL_TOP_ACTIVATE).filter(finance.STK_EL_TOP_ACTIVATE.day==i,
               finance.STK_EL_TOP_ACTIVATE.link_id <= 310002)
        temp=finance.run_query(q)
        df=pd.concat([df,temp])
    
    df_=df[['day','code','name','buy','sell']]
    df_['net']=df_.buy-df_.sell
    df_['net_s']=df_.net.apply(lambda s:1.0 if s>0 else 0.0)
    out1=pd.DataFrame()
    num=0
    for idx,group_ in df_.groupby('code'):
        if num==0:            
            temp=group_.copy()
            temp['MA']=tb.MA(temp['net_s'].values,ma_N)
            out1=temp
        else:
            temp=group_.copy()
            temp['MA']=tb.MA(temp['net_s'].values,ma_N)
            out1=pd.concat([out1,temp])
        num=num+1
    out1=out1.dropna()
    out1.day=out1.day.apply(lambda s:str(s)[:10])
    res=out1.query("day=='{var1}' & MA>=1".format(var1=tst[-2]))
    print('*************************************')
    print('连续{var1}日资金流入的个股'.format(var1=ma_N)) 
    print(res[['day','code','name']])       

# =============================================================================
# 通过港股通持仓变化 寻找连续增仓个股        
# =============================================================================
    df_1=finance.run_query(query(finance.STK_HK_HOLD_INFO).filter(finance.STK_HK_HOLD_INFO.link_id != 310005,
                         finance.STK_HK_HOLD_INFO.day==tst[0]))
    for i in tst[1:]:
        temp=finance.run_query(query(finance.STK_HK_HOLD_INFO).filter(finance.STK_HK_HOLD_INFO.link_id != 310005,
                         finance.STK_HK_HOLD_INFO.day==i))
        df_1=pd.concat([df_1,temp])
    df_1=df_1[['day','code','name','share_ratio']]
    
    num=0
    for idx,group_ in df_1.groupby('code'):
        if num==0:
            temp=group_.copy()
            temp['share_change']=group_.share_ratio.diff()
            temp['share_change_s']=temp.share_change.apply(lambda s:1.0 if s>0 else 0.0)
            temp['share_change_ma']=tb.MA(temp['share_change_s'].values,ma_M)
            out2=temp
        else:
            temp=group_.copy()
            temp['share_change']=group_.share_ratio.diff()
            temp['share_change_s']=temp.share_change.apply(lambda s:1.0 if s>0 else 0.0)
            temp['share_change_ma']=tb.MA(temp['share_change_s'].values,ma_M)
            out2=pd.concat([temp,out2])
        num=num+1
    
    out2=out2.dropna()
    out2.day=out2.day.apply(lambda s:str(s)[:10])
    res1=out2.query("day=='{var1}' & share_change_ma>=1".format(var1=tst[-2]))
    print('*************************************')
    print('连续{var1}日资金流入的个股(全样本)'.format(var1=ma_M)) 
    print(res1[['day','code','name']])       
    

