# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 14:21:10 2020
指数择时测试框架
@author: Administrator
"""

from __future__ import division
from backtest_func import *
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import talib as tb
import numpy as np
#style.use('ggplot')
from jqdatasdk import *
auth('18610039264', 'zg19491001')

# 获取价格
def stock_price(sec, period, sday, eday):
    temp = get_price(sec, start_date=sday, end_date=eday, frequency=period, fields=['open', 'close', 'high', 'low'],
                     skip_paused=True, fq='pre', count=None).reset_index() \
        .rename(columns={'index': 'tradedate'})
    temp['stockcode'] = sec
    return temp


if __name__ == '__main__':

    start='2010-01-01'
    end='2020-09-01'
    period = '1d'
    code ='000300.XSHG'
    strategy = 'single_line'
    n = 1440
    # 单均线策略
    N1_lst=[5,10,20,30,60]  # 短均线参数
    fee = 0.004
    symble_lst = 'A股指数'

    df_lst = []
    lst = []
    state_lst = []
    
# =============================================================================
# 原来的数据提取    
# =============================================================================
#    group = get_huobi_ontime_kline('btcusdt','60min', start, end)[2].reset_index() \
#              .rename(columns={'date':'date_time'})
#    
#
#    group_day = get_huobi_ontime_kline(symble_lst,'1day',start,end)[2].reset_index()[
#        ['date','high','low','close']]\
#        .rename(columns={'date':'date_time'}) \
#        .assign(date_time=lambda df: df.date_time.apply(lambda x: x[:10] + ' 00:00:00'))\
#        .sort_values(['date_time'])
    
    group = stock_price(code, period, start, end)
    
    group = group.rename(columns={'tradedate':'date_time'}) \
                 .assign(date_time=lambda df: df.date_time.apply(lambda x: str(x)[:10]))\
                 .sort_values(['date_time'])

  
    for N1 in N1_lst:                     
        print(symble_lst)
        method = 'single_line' +'_'+str(N1)            
        print(method)
        signal_lst = []
        trad_times = 0
        if len(group) > N1:
            net = 1
            net_lst = []
            group_ = group.copy()

            group_=group_.assign(MA1=lambda df:tb.MA(df['close'].values,N1))\
                         .assign(close_1=lambda df:df.close.shift(1))\
                         .assign(MA1_1=lambda df:df.MA1.shift(1))
                            
            group_=group_.dropna()
            group_.index=range(len(group_))
            position = 0
            high_price_pre = 10000000
            # signal_row = []
            # stock_row = []
            for idx, _row in group_.iterrows():
                if (position == 0) & (_row.close_1 > _row.MA1_1):
                    position = 1
                    s_time = _row.date_time
                    cost = _row.open
                    hold_price = []
                    high_price = []
                    hold_price.append(cost)
                    high_price.append(cost)
                    net = net * _row.close / cost
               
                elif position == 1:
                    if _row.close_1 < _row.MA1_1:
                        position = 0
                        trad_times += 1
                        high_price.append(_row.high)
                        e_time = _row.date_time
                        s_price = _row.open                
                        ret = s_price / cost - 1
                        signal_row = []
                        signal_row.append(s_time)
                        signal_row.append(e_time)
                        signal_row.append(cost)
                        signal_row.append(s_price)
                        signal_row.append(ret - fee)
                        signal_row.append(max(high_price) / cost - 1)
                        signal_row.append(len(hold_price))
                        net = net * s_price / _row.close_1 * (1 - fee)
                        signal_lst.append(signal_row)
                    else:
                        high_price.append(_row.high)
                        hold_price.append(_row.close)
                        net = net * _row.close / _row.close_1

                net_lst.append(net)

            ann_ROR = annROR(net_lst, n)
            total_ret = net_lst[-1]
            max_retrace = maxRetrace(net_lst)
            sharp = yearsharpRatio(net_lst, n)
            plt.plot(net_lst)
            signal_state = pd.DataFrame(signal_lst,
                                        columns=['s_time', 'e_time', 'b_price', 's_price', 'ret',
                                                 'max_ret', 'hold_day']) \
                .assign(method=method)
            signal_state.to_csv('result/signal_sl_' + str(n) + '_' + method + '.csv')
            # df_lst.append(signal_state)
            win_r, odds, ave_r, mid_r = get_winR_odds(signal_state.ret.tolist())
            win_R_3, win_R_5, ave_max = get_winR_max(signal_state.max_ret.tolist())
            state_row = []
            state_row.append(symble_lst)
            state_row.append(n)
            state_row.append(win_r)
            state_row.append(odds)
            state_row.append(total_ret - 1)
            state_row.append(ann_ROR)
            state_row.append(sharp)
            state_row.append(max_retrace)
            state_row.append(len(signal_state))
            state_row.append(ave_r)
            state_row.append(signal_state.hold_day.mean())
            state_row.append(mid_r)
            state_row.append(win_R_3)
            state_row.append(win_R_5)
            state_row.append(ave_max)

            state_row.append(N1)
            
#                        state_row.append(K2)
            state_lst.append(state_row)
            # print('胜率=', win_r)
            # print('盈亏比=', odds)
            # print('总收益=', total_ret - 1)
            # print('年化收益=', ann_ROR)
            # print('夏普比率=', sharp)
            # print('最大回撤=', max_retrace)
            # print('交易次数=', len(signal_state))
            # print('平均每次收益=', ave_r)
            # print('平均持仓周期=', signal_state.hold_day.mean())
            # print('中位数收益=', mid_r)
            # print('超过3%胜率=', win_R_3)
            # print('超过5%胜率=', win_R_5)
            # print('平均最大收益=', ave_max)
            # print('参数=', method)

    signal_state = pd.DataFrame(state_lst, columns=['symble', 'tm', 'win_r', 'odds', 'total_ret', 'ann_ret', 'sharp',
                                                    'max_retrace', 'trade_times', 'ave_r', 'ave_hold_days', 'mid_r',
                                                    'win_r_3', 'win_r_5', 'ave_max', 'n1'])
    print(signal_state)
    signal_state.to_csv('result/state_sl.csv')