#coding:utf-8
'''
Created on 2016-3-28

@author: li
'''

import time, threading
import tushare as ts
import datetime
import pandas as pd

data_thread_running=True

index_code_used=['000001', '399001','399005','399006','000300','000016']
time_trade_zq_dict={'begintime':['9:15','13:01'], 'endtime':['11:30','15:31']}
#mutex = threading.Lock()
stocks_kline_data_dict={}

dict_stock_index = {}
#dict_stock_code = []
_t=None
#local_data = threading.local()

_dict_callback={}

def get_stock_index_data():
    if not dict_stock_index.has_key('data'):

        for i in range(30):
            if dict_stock_index.has_key('data'):
                break;
            else:
                time.sleep(2)
    return dict_stock_index['data']

def _get_stock_realtime_data():
    df=pd.DataFrame()
    df = ts.get_today_all()
    ky=df.keys()
    df=df.set_index('code').T.to_dict('list')
    return df,ky
def _fill_stock_code_list( realtime_data):
    stock_code_list=[]
    for code, val in realtime_data.items():
        code_name={}
        code_name['code']=code
        code_name['name']=val[0]
        stock_code_list.append(code_name)
    return stock_code_list
# 新线程执行的代码:
