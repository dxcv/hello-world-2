#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 18:23:29 2019
基金监控面板 进行基金组合的分析和统计
@author: yeecall
"""

from __future__ import division
import pandas as pd
import numpy as np
# import os
# py_path=r'/Users/yeecall/Documents/mywork/joinquant_data/基金评价'
# os.chdir(py_path)
import matplotlib.pyplot as plt
import math
import datetime
#from statsmodels import regression
#import statsmodels.api as sm
from jqdatasdk import *
auth('18610039264','zg19491001')


# 提取基金净值数据
def fund_value(start_day,code):
      q=query(finance.FUND_NET_VALUE.code,
              finance.FUND_NET_VALUE.day,
              finance.FUND_NET_VALUE.sum_value).filter(finance.FUND_NET_VALUE.code==code,
              finance.FUND_NET_VALUE.day> start_day)
      df=finance.run_query(q)
      return(df)
# 价格转净值 起初资产赋权
def priceTovalue(df,w):
      f=df.iloc[0,:].tolist()
      for i in range(len(f)):
            df.iloc[:,i]=df.iloc[:,i]*w[i]/f[i]
      df['portfolio']=df.sum(axis=1)
      return df
# 价格转净值 日变化率赋权            
def priceTovalue_(df,w):
     chg = df.diff()/df
     for i in range(len(chg.columns)):
           chg.iloc[:,i]=chg.iloc[:,i]*w[i]
     chg = chg.fillna(0)
     chg['portfolio_chg']=chg.sum(axis=1)
     chg['portfolio'] = (1 + chg['portfolio_chg']).cumprod()
     chg.to_excel('test.xls')
     return chg
           
def maxRetrace(list):
          '''
          :param list:netlist
          :return: 最大历史回撤
          '''
          Max = 0.0001
          for i in range(len(list)):
              if 1 - list[i] / max(list[:i + 1]) > Max:
                  Max = 1 - list[i] / max(list[:i + 1])
      
          return Max
           
def yearsharpRatio(netlist):
    row = []
    for i in range(1, len(netlist)):
        row.append(math.log(netlist[i] / netlist[i - 1]))
    return np.mean(row) / np.std(row) * math.pow(252, 0.5)
      
def annROR(netlist,days):
    '''
    :param netlist:净值曲线
    :return: 年化收益
    '''
    return math.pow(netlist[-1] / netlist[0], 365/days) - 1

# Var 
def Var(lst,a=0.01):
    '''
    :param list:netlist
    :return: 平均回撤率
    '''
    from scipy.stats import norm

    llst=[np.log(i) for i in lst]
    llst_0=llst[:-1]
    llst_1=llst[1:]
    lr=list()
    for i in range(len(llst_1)):
          lr.append(llst_1[i]-llst_0[i])
    m=np.mean(lr)
    d=np.std(lr)
    var=norm.ppf(a)*d+m
    return(var)

def index_(code, start, end):
    df = get_price(code, start_date=start, end_date=end)
    
    return df

if __name__=='__main__':
      # fund=['180012','519732','110011','519736','000751','070032'] # 一期股票和混合基金的代码
      # fund=['100072','320021','000171'] #债券
      fund=['000597','519732','519738']# 混合型
      #fund=['000751','110022','164205','000577','160222']# 股票型
      #w=[1/len(fund) for i in fund] #权重
      # w=opts['x']
      # w=[0.18,0.43,0.06,0.01,0.31,0.01] # 第一次的权重
      w=[0.50,0.00,0.50] #债券
      #w=[0.24,0.10,0.10,0.16,0.4] #混合型
      # w=[0.15,0.35,0,0.05,0.45] #股票型
      # money=5000
      # everyone=[money*i for i in w]
      # fund = ['000751','110022','164205','100072','320021','000171','000410','519732','519697']
      # w = [0.22,0.1,0.02,0.22,0.02,0.02,0.22,0.09,0.09]
      # w = [0.10,0.10,0.10,0.10,0.10,0.10,0.14,0.13,0.13]
      flag=0
      start='2015-01-01'
      
      end = datetime.datetime.today()
      day_ = (end - datetime.datetime.strptime(start, '%Y-%m-%d')).days
      # end=str(end)[:10]
      
#      end='2018-12-31'
      
      # 合成基金净值数据
      print("生成净值数据")
      print(fund[0])
      df=fund_value(start,fund[0])
      for i in range(len(fund))[1:]:
            print(fund[i])
            df=pd.concat([df,fund_value(start,fund[i])])
      df.code=df.code.apply(lambda s:str(s)+'.jj')
      df=pd.pivot_table(df,index='day',columns='code',values='sum_value')
      fund_ = [str(s)+'.jj' for s in fund]
      df = df[fund_]
      if flag==0:            
      # 生成净值
            ret=list()
            fundvalue=priceTovalue(df,w)
            fundvalue=fundvalue.dropna()
      
            fundvalue.loc[:,'portfolio'].plot(title='TYPE OF HYBRID')
            # fundvalue.to_excel('fundvalue.xls')
            index300 = index_('000300.XSHG', '2015-01-01', '2020-01-23')
            # index300 = index300.loc[:,'close']
            # index300['chg'] = index300['close'].diff() / index300['close']
            # index300['net_300'] = (1 + index300['chg']).cumprod()
            index300['net_300'] = index300['close']/index300['close'].tolist()[0]
            index300['net_300'].plot()
            net=fundvalue.portfolio.tolist()
            net_252=net[-252:]
            days=fundvalue.index.tolist()
            s=str(days[0])[:10]
            e=str(days[-1])[:10]
           #近7天收益率：          
            yoy_7=(net[-1]-net[-7])/net[-7]
           #据最高净值回撤：
            H_Retrace=(net[-1]-max(net))/max(net)
           #近1年收益率：
            yoy_252=(net[-1]-net[-252])/net[-252]
             
           #近一年最大回撤：
            maxRet_252=maxRetrace(net_252)            
           #近一年夏普比率：
            sharp_252=yearsharpRatio(net_252)            
           #近一年VAR:
            var_252=Var(net_252)
            # 年化收益率
            annror = annROR(net,day_)
            #近一年最大回撤：
            maxRet=maxRetrace(net)            
           #近一年夏普比率：
            sharp=yearsharpRatio(net)            
           #近一年VAR:
            var=Var(net)
            
            ret.append(s)
            ret.append(e)
            ret.append(yoy_7)
            ret.append(H_Retrace)
            ret.append(yoy_252)
            ret.append(maxRet_252)
            ret.append(sharp_252)
            ret.append(var_252)
            ret.append(annror)
            ret.append(maxRet)
            ret.append(sharp)
            ret.append(var)
            result=pd.DataFrame(ret)
            result['说明']=['开始日期','结束日期','近七天收益率','高水位回撤','近1年收益率','近一年最大回撤','近一年夏普','近一年VAR','历史年化收益率','历史最大回撤','历史夏普率','VAR']
            result.columns=['值','说明']
            result = result[['说明' , '值']]
            result.to_excel('result.xls', encoding='gbk')
            print(result)
            
# =============================================================================
#  基于MTP 计算夏普最优参数     
# =============================================================================
      else:
            # Calculate the annualized mean returns and covariance matrix
            df=df.reset_index()
            df.day=df.day.apply(lambda s:str(s)[:10])
#            df=df.query("day<'{var}'".format(var=end))
            df=df.set_index('day',drop=True)
            rets = np.log(df / df.shift(1)).dropna()
            annual_mean_rets = rets.mean() * 252
            # The covrance of random walk is in proportion to time
            annual_cov_rets = rets.cov() * 252
            
            def port_ret(weights):
                  return np.sum(annual_mean_rets*weights)
            
            def port_vol(weights):
                  return np.sqrt(np.dot(weights.T, np.dot(annual_cov_rets, weights)))
      
            import scipy.optimize as sco
            # Function to be minimized
            def min_func_sharpe_ratio(weights):
                return -port_ret(weights) / port_vol(weights)
            
            noa = len(df.columns)
            # Equality constraint
            cons = ({'type':'eq', 'fun': lambda x: np.sum(x)-1})
            
            # Bounds for the parameters
            bnds = tuple((0, 1) for x in range(noa))
            
            # Initial parameters
            eweights = np.array(noa * [1. /noa,])
            
            opts = sco.minimize(min_func_sharpe_ratio, eweights, method='SLSQP',
                         bounds=bnds, constraints=cons)
            w=opts['x']
            print(opts['x'])
            df.corr().to_csv('result/value_corr.csv')
            
