# coding=utf-8
'''
Created on 7.9, 2018

@author: fang.zhang
'''
from __future__ import division
import numpy as np
import math
#rom ConfigDB import *
import pandas as pd
#import MySQLdb
#from sqlalchemy import create_engine
import time
#import sqlalchemy
import os
import talib

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'


class MysqlDBConnector(object):
    def __init__(self, dbKey=None):
        if dbKey is None:
            self.connPara = MssqlConnParaMap['local']
        else:
            self.connPara = MssqlConnParaMap[dbKey]
        return

    def build_database_connection(self):
        try:
            conn = MySQLdb.connect(host=self.connPara['server'], user=self.connPara['user'],
                                   passwd=self.connPara['password'], db=self.connPara['database'])
        except MySQLdb.DatabaseError  as e:
            print("Can not connect to server")
        return conn

    def build_alchemy_connection(self):
        try:
            engine = create_engine(
                'mysql+mysqldb://' + self.connPara['user'] + ':' + self.connPara['password'] + '@' + self.connPara[
                    'server'] + ':3306/' + self.connPara['database'])
        except engine.Error  as e:
            print("Can not connect to server")
        return engine

    def write_data_to_db(self, datadf, TableName, mode=1):
        engine = self.build_alchemy_connection()
        if mode == 2:
            datadf.to_sql(TableName, engine, if_exists='replace', index=False, index_label=None, chunksize=None,
                          dtype={'data_type': sqlalchemy.types.String(32), 'last_time': sqlalchemy.TIMESTAMP})
        elif mode == 3:
            datadf.to_sql(TableName, engine, if_exists='append', index=False, index_label=None, chunksize=None,
                          dtype=None)
        else:
            datadf.to_sql(TableName, engine, if_exists='fail', index=False, index_label=None, chunksize=None,
                          dtype=None)

    def get_data_from_query(self, stmt):
        conn = self.build_database_connection()
        df = pd.read_sql(stmt, conn)
        conn.close()
        return df

    def delete_data_from_query(self, stmt):
        conn = self.build_database_connection()
        cursor = conn.cursor()
        ret = cursor.execute(stmt)
        cursor.close()
        conn.close()
        return ret

    def get_query_stmt(self, tableName, colNames, constraints, orderby):
        stmt = 'select '
        for col in colNames:
            stmt = stmt + col + ','
        stmt = stmt[0:len(stmt) - 1]
        stmt = stmt + ' from ' + tableName
        if constraints is None:
            stmt = stmt + ''
        else:
            stmt = stmt + constraints

        if orderby is None:
            return stmt
        else:
            stmt = stmt + ' order by '
            stmt = stmt + orderby
            return stmt

    def get_data(self, tableName, colNames, constraints, orderby):
        assert colNames is not None
        try:
            conn = self.build_database_connection()
            stmt = self.get_query_stmt(tableName, colNames, constraints, orderby)
            cursor = conn.cursor()
            t = time.time()
            cursor.execute(stmt)
            df = pd.DataFrame.from_records(cursor.fetchall())
            if len(df) > 0:
                df.columns = colNames
        except MySQLdb.Error as e:
            conn.rollback()
            message = "SqlServer Error %d: %s" % (e.args[0], e.args[1])
            print(message)
        finally:
            cursor.close()
            conn.close()
        print('time elapsed for this oracle query: ', time.time() - t)
        return df

    def update_data_to_db(self, TableName, chng_cloname, chng_clovalue, cloname_lst, value_lst):
        connect = MySQLdb.connect(host=self.connPara['server'], port=3306, user=self.connPara['user'],
                                  passwd=self.connPara['password'], db=self.connPara['database'],
                                  charset='utf8')
        cur = connect.cursor()
        sql1 = 'update ' + TableName + ' set ' + chng_cloname + ' = %s where ('
        sql2 = ''
        for colname in cloname_lst:
            sql2 = sql2 + colname + '= %s)&('
        sql = sql1 + sql2[:-2]
        lst = []
        lst.append(chng_clovalue)
        lst.extend(value_lst)
        cur.execute(sql, lst)
        connect.commit()
        connect.close()


def get_symble_id(exchange, codecoin, basecoin):
    stmt = 'select id from sys_exchange where name like ' + '\'' + exchange + '\''
    csd_hq = MysqlDBConnector('production')
    exchange_id = csd_hq.get_data_from_query(stmt).iat[0, 0]
    symbol_stmt = 'select id from sys_exsymbol where exchangeid = ' + '\'' + str(exchange_id) + '\'' + \
                  ' and basecoin like ' + '\'' + codecoin + '\'' + ' and quotecoin like ' + '\'' + basecoin + '\''
    data = csd_hq.get_data_from_query(symbol_stmt)
    return data.iat[0, 0]


def get_symble_id_all(exchange):
    stmt = 'select id, symbol from sys_exchange where name like ' + '\'' + exchange + '\''
    csd_hq = MysqlDBConnector('production')
    exchange_id = csd_hq.get_data_from_query(stmt).iat[0, 0]
    symbol_stmt = 'select id from sys_exsymbol where exchangeid like ' + '\'' + str(exchange_id) + '\''
    csd_hq = MysqlDBConnector('production')
    data = csd_hq.get_data_from_query(symbol_stmt)
    return data


def get_market_kline(exsymbol_id, period, s_time, e_time):
    s_time = str(int(time.mktime(time.strptime(s_time, '%Y-%m-%d %H:%M:%S'))))
    e_time = str(int(time.mktime(time.strptime(e_time, '%Y-%m-%d %H:%M:%S'))))
    stmt = 'select close, amount, count, high, low, open, period, tickid, volume  from market_kline where exsymbolid ' \
           '= ' + '\'' + str(exsymbol_id) + '\'' + ' and period like ' + '\'' + period + '\'' + ' and tickid between ' \
           + '\'' + s_time + '\'' + ' and ' + '\'' + e_time + '\'' + ' order by tickid'
    csd_hq = MysqlDBConnector('production')
    data = csd_hq.get_data_from_query(stmt).assign(
        date_time=lambda df: df.tickid.apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x))))
    return data


def get_market_kline_more_exsymbol(exsymbol_id_lst, period, s_time, e_time):
    s_time = str(int(time.mktime(time.strptime(s_time, '%Y-%m-%d %H:%M:%S'))))
    e_time = str(int(time.mktime(time.strptime(e_time, '%Y-%m-%d %H:%M:%S'))))

    const1 = 'select close, amount, count, high, low, open, period, tickid, volume, exsymbolid from market_kline ' \
             'where exsymbolid in ('
    if exsymbol_id_lst is not None:
        for i in range(len(exsymbol_id_lst)):
            const1 = const1 + '\'' + str(exsymbol_id_lst[i]) + '\'' + ','
    const1 = const1[:-1] + ') and period like ' + '\'' + period + '\'' + ' and tickid between ' \
        + '\'' + s_time + '\'' + ' and ' + '\'' + e_time + '\'' + ' order by tickid'
    csd_hq = MysqlDBConnector('production')
    data = csd_hq.get_data_from_query(const1).assign(
        date_time=lambda df: df.tickid.apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x))))
    return data


def get_kline(exchange, symbol, period, s_time, e_time):
    s_time = str(int(time.mktime(time.strptime(s_time, '%Y-%m-%d %H:%M:%S'))))
    e_time = str(int(time.mktime(time.strptime(e_time, '%Y-%m-%d %H:%M:%S'))))
    stmt = 'select close, amount, high, low, open, period, tickid, volume, symbol  from kline where exchange ' \
           '= ' + '\'' + exchange + '\'' + ' and period like ' + '\'' + period + '\'' + ' and symbol like ' + '\'' + \
           symbol + '\'' + ' and tickid between ' + '\'' + s_time + '\'' + ' and ' + '\'' + e_time + '\'' + \
           ' order by tickid'
    csd_hq = MysqlDBConnector('production')
    data = csd_hq.get_data_from_query(stmt).assign(
        date_time=lambda df: df.tickid.apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x))))
    return data


def get_kline_period(exchange, period, s_time, e_time):
    s_time = str(int(time.mktime(time.strptime(s_time, '%Y-%m-%d %H:%M:%S'))))
    e_time = str(int(time.mktime(time.strptime(e_time, '%Y-%m-%d %H:%M:%S'))))
    stmt = 'select close, amount, high, low, open, period, tickid, volume from kline where exchange ' \
           '= ' + '\'' + exchange + '\'' + ' and period like ' + '\'' + period + '\'' + \
           ' and tickid between ' + '\'' + s_time + '\'' + ' and ' + '\'' + e_time + '\'' + ' order by tickid'
    csd_hq = MysqlDBConnector('production')
    data = csd_hq.get_data_from_query(stmt).assign(
        date_time=lambda df: df.tickid.apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x))))
    return data


def get_kline_more_exsymbol(exchange, symbol_lst, period, s_time, e_time):
    s_time = str(int(time.mktime(time.strptime(s_time, '%Y-%m-%d %H:%M:%S'))))
    e_time = str(int(time.mktime(time.strptime(e_time, '%Y-%m-%d %H:%M:%S'))))
    const1 = 'select close, amount, high, low, open, period, tickid, volume, symbol from kline where exchange ' \
             '= ' + '\'' + exchange + '\'' + ' and symbol in ('
    if symbol_lst is not None:
        for i in range(len(symbol_lst)):
            const1 = const1 + '\'' + symbol_lst[i] + '\'' + ','
    const1 = const1[:-1] + ') and period like ' + '\'' + period + '\'' + ' and tickid between ' \
        + '\'' + s_time + '\'' + ' and ' + '\'' + e_time + '\'' + ' order by tickid'
    csd_hq = MysqlDBConnector('production')
    data = csd_hq.get_data_from_query(const1).assign(
        date_time=lambda df: df.tickid.apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x))))
    return data


def get_winR_odds(ret_lst):
    win_lst = [i for i in ret_lst if i > 0]
    loss_lst = [i for i in ret_lst if i < 0]
    win_R = 0
    odds = 1
    ave = 0
    mid_ret = 0
    if len(win_lst) + len(loss_lst) > 0:
        win_R = len(win_lst) / (len(win_lst) + len(loss_lst))
        ave = (sum(win_lst) + sum(loss_lst)) / (len(win_lst) + len(loss_lst))
        odds = 10
        if len(win_lst) == 0:
            win_lst = [0]
        if len(loss_lst) > 0:
            odds = - np.mean(win_lst) / np.mean(loss_lst)
        win_lst.extend(loss_lst)
        mid_ret = np.percentile(win_lst, 50)
    return win_R, odds, ave, mid_ret


def get_winR_max(ret_lst):
    win_lst_3 = [i for i in ret_lst if i > 0.03]
    loss_lst_3 = [i for i in ret_lst if i < 0.03]
    win_lst_5 = [i for i in ret_lst if i > 0.05]
    loss_lst_5 = [i for i in ret_lst if i < 0.05]
    win_R_3 = 0
    win_R_5 = 0
    ave_max = 0
    if len(win_lst_3) + len(loss_lst_3) > 0:
        win_R_3 = len(win_lst_3) / (len(win_lst_3) + len(loss_lst_3))
        ave_max = (sum(win_lst_3) + sum(loss_lst_3)) / (len(win_lst_3) + len(loss_lst_3))
    if len(win_lst_5) + len(loss_lst_5) > 0:
        win_R_5 = len(win_lst_5) / (len(win_lst_5) + len(loss_lst_5))

    return win_R_3, win_R_5, ave_max


def maxRetrace(list):
    '''
    :param list:netlist
    :return: 最大历史回撤
    '''
    Max = 0
    for i in range(len(list)):
        if 1 - list[i] / max(list[:i + 1]) > Max:
            Max = 1 - list[i] / max(list[:i + 1])

    return Max


def annROR(netlist, n):
    '''
    :param netlist:净值曲线
    :return: 年化收益
    '''
    return math.pow(netlist[-1] / netlist[0], 365 * 1440 / len(netlist) / n) - 1


def daysharpRatio(netlist):
    row = []
    for i in range(1, len(netlist)):
        row.append(math.log(netlist[i] / netlist[i - 1]))
    return np.mean(row) / np.std(row)


def yearsharpRatio(netlist, n):
    row = []
    for i in range(1, len(netlist)):
        row.append(math.log(netlist[i] / netlist[i - 1]))
    return np.mean(row) / np.std(row) * math.pow(365 * 1440/n, 0.5)


def get_min_n_from_period(period):
    if (period == '5m') | (period == '5min'):
        n = 5
    if (period == '15m') | (period == '15min'):
        n = 15
    if (period == '30m') | (period == '30min'):
        n = 30
    if (period == '60min') | (period == '1hour') | (period == '1h'):
        n = 60
    if (period == '240m') | (period == '4hour') | (period == '4h'):
        n = 240
    if (period == '1day') | (period == '1d'):
        n = 1440
    return n


if __name__ == '__main__':

    data = get_kline('BIAN', 'btcusdt', '1m', '2018-08-17 00:00:00', '2018-10-01 00:00:00')
    print(data)
    data.to_csv('data/btcusdt_1m.csv')


