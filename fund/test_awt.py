# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 13:21:22 2020
爱玩特智能-测试题
@author: Administrator
"""
import pandas as pd
import statsmodels.api as sm
import math

def cal_annror(df):
    tmp = df.dropna()
    ret_ask = list()
    for idx,group_ in tmp.groupby('ID'):
        print('客户id' + str(idx))
        tmp_lst = list()
        annror = group_['yield'].mean() * 252
        tmp_lst.append(idx)
        tmp_lst.append(annror)
        ret_ask.append(tmp_lst)   
    all_customers_annror = pd.DataFrame(ret_ask,columns=['ID','annror'])
    return all_customers_annror
    

if __name__ == '__main__':
    # 读取原始数据
    data = pd.read_excel("C:/JQ/testData.xlsx",encoding='gbk')
    
    # 问题1 画时间和用户数的走势图
    customers = data.groupby('As_Of_Date')['ID'].count()
    
    customers.to_excel("C:/JQ/answer1.xls",encoding='gbk')
    customers.plot(title='quantity_of_customers')
    
    # 问题2 统计每个客户每日的投资回报率
    print("计算每个客户日回报率")
    ret = list()
    for idx,group_ in data.groupby('ID'):
        print('客户id' + str(idx))
        tmp = group_.sort_values(by='As_Of_Date')
        tmp = tmp[['As_Of_Date','ID','Total_account']]
        tmp['yield'] = tmp.Total_account.diff() / tmp.Total_account.shift(1)
        ret.append(tmp)
    all_customers_daily_yield = pd.concat(ret)
    all_customers_daily_yield.to_excel("C:/JQ/answer2.xls",encoding='gbk')
    
    # 问题3  每个客户年化回报率，年化波动率，夏普比率
    print("计算每个客户描述性统计")
    all_customers_daily_yield_ask3 = all_customers_daily_yield.dropna()
    ret_ask3 = list()
    for idx,group_ in all_customers_daily_yield_ask3.groupby('ID'):
        print('客户id' + str(idx))
        tmp_lst = list()
        annror = group_['yield'].mean() * 252
        annvol = group_['yield'].std() * (252**0.5)
        sharp = annror / annvol
        tmp_lst.append(idx)
        tmp_lst.append(annror)
        tmp_lst.append(annvol)
        tmp_lst.append(sharp)
        ret_ask3.append(tmp_lst)   
    all_customers_indicators = pd.DataFrame(ret_ask3,columns=['ID','annror','annvol','sharp'])
    all_customers_indicators.to_excel("C:/JQ/answer3.xls",encoding='gbk') 
    
    # 问题4  计算全样本前95%日期中最后一天，和后5%的最后一天
    ask_4 = customers.reset_index()
    date_lst = ask_4.As_Of_Date.tolist()
    # 95%的最后一个事件的序号
    percent95 = math.floor(0.95*len(date_lst))
    # 全样本前95%的最后一天
    percent95_all_customers = date_lst[percent95-1]
    # 全样本后5%的最后一天
    percent5_all_customers = date_lst[-1]
    
    # 问题 4 对前95%的样本 每个客户收益率 与 后5%的样本 每个客户收益率进行线性回归
    ask_5_95_f = all_customers_daily_yield[all_customers_daily_yield['As_Of_Date']<=percent95_all_customers]
    ask_5_95_b = all_customers_daily_yield[all_customers_daily_yield['As_Of_Date']>percent95_all_customers]
    annror_95_f = cal_annror(ask_5_95_f) # 计算前95%的年化收益率
    annror_5_b = cal_annror(ask_5_95_b) # 计算后5%的年化收益率
    # 进行线性回归
    ask_5 = pd.merge(annror_95_f,annror_5_b,on='ID')
    x = ask_5['annror_x']
    y = ask_5['annror_y']
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    model.summary2()
    
        
        
        
    
    
    
    