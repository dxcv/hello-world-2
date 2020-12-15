# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 14:52:24 2020

@author: Administrator
"""

import pandas as pd

# tm = pd.read_csv("test.csv",encoding='gbk')
# tm['label'].value_counts()

tm1 = pd.read_excel("fund_info2.xlsx",encoding='gbk',index_col=0)
tm1.head()

# 基金统计
tm11 = tm1.drop_duplicates(subset=['MainCode'])

col_lst = ['InnerCode','基金公司名称','基金一级分类','近1年收益率','近1年收益率排名'
           ,'近3年收益率','近3年收益率排名','近1年最大回撤','近1年最大回撤排名',
           '近3年最大回撤','近3年最大回撤排名','近1年夏普率','近1年夏普率排名',
           '近3年夏普率','近3年夏普率排名']

test = tm11[col_lst]
test = test[test['基金公司名称'] == '交银施罗德基金管理有限公司']
test =test[test['基金一级分类'].isin(['股票型','混合型','债券型'])]
ss = test.groupby('基金一级分类').mean().reset_index()
ss.to_excel("result_sb.xls",encoding='gbk')
# 基金经理
tm12 = tm1.drop_duplicates(subset=['Name','基金一级分类'])
test2 = tm12[tm12['基金公司名称'] == '交银施罗德基金管理有限公司']
col_lst = ['Name','基金经理近1年收益率','manager_ir_1_rank','基金经理近3年收益率','manager_ir_3_rank'
           ,'基金经理近1年最大回撤率','manager_md_1_rank','基金经理近3年最大回撤率','manager_md_3_rank',
           '基金经理近1年夏普率','manager_s_1_rank','基金经理近3年夏普率','manager_s_3_rank']





