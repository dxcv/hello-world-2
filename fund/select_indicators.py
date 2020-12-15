# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 14:45:42 2020
对指标体系进行指标筛选
Step 1： 数据整理，分别计算每只基金的各种指标的月频指标，未来1年的MRAR（2）值以及基于未来1年的MRAR（2）值的目标评级；
Step 2： 计算各种指标的IC，IR值，根据筛选规则进行指标一次筛选；
Step 3： 计算各个指标的方差值和协相关矩阵，剔除常量指标和相关性大的指标；
Step 4 :  分别对每个指标值与目标评级结果进行卡方检验，剔除p>0.05的指标；
@author: lufeiepng
"""

from __future__ import division
import pandas as pd
from MySQLdb import connect, cursors
import configparser
import socket
import numpy as np
import time
from sqlalchemy import create_engine  
from datetime import datetime
from dateutil.relativedelta import relativedelta



if __name__ == '__main__':
    # 读取指标数据
    df = pd.read_csv('data/funds_indicators.csv', index_col=0)
