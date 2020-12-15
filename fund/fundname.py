# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 17:18:12 2020
下载全部基金名称和代码
@author: Administrator
"""

import pandas as pd
from jqdatasdk import *

auth('18610039264', 'zg19491001')

def fund_find(operate_mode, underlying_asset_type):
    q = query(finance.FUND_MAIN_INFO).filter(finance.FUND_MAIN_INFO.operate_mode_id == operate_mode,
                                             finance.FUND_MAIN_INFO.underlying_asset_type_id == underlying_asset_type)
    df = finance.run_query(q)
    print('一共' + str(len(df)) + '只基金')
    return (df)

if __name__ == '__main__':
    operate_mode_id = [401001, 401006]
    underlying_asset_type_id = [402001, 402003, 402004]
    # fund_id 为符合条件的基金名单
    ret = list()
    for i in operate_mode_id:
        for j in underlying_asset_type_id:
            tmp = fund_find(i, j)
            ret.append(tmp)
    fund_id = pd.concat(ret)

    fund_id['end_date'] = fund_id['end_date'].fillna(1)
    fund_id = fund_id[fund_id['end_date'] == 1]
    fund_id.to_excel("fund_name.xls",encoding='gbk')
    
    