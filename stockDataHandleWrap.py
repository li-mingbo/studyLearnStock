#coding:utf-8
'''
Created on 2016-1-19

@author: li
'''

import tushare as ts
from numpy import *
import pandas as pd
import os 
import csv
import time
import datetime
import pickle as pk
#from train import litestsl
import litestcy as litestsl
#import matplotlib.pyplot as plt
from pandas.core.datetools import thisBMonthEnd
import string
from io import StringIO as sio
import ConstantDefine
import DataSource

mode_targets = {"normal":0, "up":1, "down":2}


def strToDataFrame(inStr):
    s = sio.StringIO()
    s.write(inStr)
    s.seek(0)        
    df=pd.read_csv(s, skip_blank_lines=True,sep='\t')
    return df


class HandleWrapClass(object):
    '''
    classdocs
    '''


    def __init__(self, params):
        '''
        Constructor
        '''
        pass

    count_each_vl=15
    normalization_mode=2
    max_data_train = 4
    m_max_need=-0.00001
    labels_count=5
    mode_target = 0
    zf_max_rate = 5

    mode_add_ma240 = False
    mode_use_price = True

    def save_hist_data(self, fm, path_name, stock_id,period_id):
        if fm is None:
            return
        pathName=path_name+period_id
        if (os.path.isdir(pathName)==False):
            os.mkdir(pathName)
        #os.mknod(pathName+stock_id+".csv")
        fm.to_csv(pathName+'/'+stock_id+".csv", "\t", '')
        
    def ma_col(self, fm, in_col_name, out_col_name, count):
        fm[out_col_name] = pd.rolling_mean(fm[in_col_name], count)
        bakdt= fm[out_col_name].copy()
        for i in range(count-1, bakdt.shape[0]):
            fm[out_col_name].iat[i-(count-1)]=bakdt.iloc[i]
        fm=fm.iloc[0:bakdt.shape[0]-count]
        return fm
    
    def transformData(self, fm):
        #fm2=pd.DataFrame(fm,index=fm.index)
        fm2=pd.DataFrame(index=fm.index)
        ky=fm.keys()
        clms=['date', 'open', 'high', 'low', 'close', 'volume']
        for ky_itm in clms:
            if ky_itm in ky:
                fm2[ky_itm]=fm[ky_itm]
        if clms[0] in ky:
            fm2=fm2.sort( columes=[clms[0]],axis=0, ascending=False)
        fm2=self.ma_col(fm2, 'close', 'ma5', 5)
        pp=fm2.pop('ma5')
        fm2.insert(2,'ma5', pp)
            
        fm3=fm2.sort_index( axis=0, ascending=False)#columns='date',
        return fm3
    
    def transformDataAll(self, fm, stock_id, period_id, drop_head_rows=True):
        fm2=pd.DataFrame(fm,index=fm.index)
        sp=fm2.shape[0]
        sp=fm2.shape[1]
        fm2['ochl_1']=fm2['close']
        for i in range(1, fm2['ochl_1'].shape[0]):
            fm2['ochl_1'].iat[i]=fm2['close'].iloc[i-1]
        sp1=fm2.shape[0]
        sp2=fm2.shape[1]
        #fm2.query('ochl')
        
        md=0
        ma_list = [2, 3, 4, 5]
        mx = max(ma_list)
        i_begin = sp2
        i_curr = 0;
        for ma in ma_list:
            str_col = 'ochl_' + str(ma)
            fm2[str_col] = fm2['close'].copy()
            for i in range(ma, fm2[str_col].shape[0]):
                fm2[str_col].iat[i] = fm2['close'].iloc[i - ma]
            i_curr += 1

        if drop_head_rows:
            fm2=fm2.iloc[mx+1:]
        self.labels_count = mx
        
        if self.max_data_train == 4:
            fm2=fm2.drop('high',axis=1)
            fm2=fm2.drop('low',axis=1)

        if 'date' in fm2.keys():
            fm2=fm2.drop('date',1)
        
        fm2=self.data_normalization(fm2, stock_id, period_id)
    
        return fm2
    def after_read_hist_data(self, fm):
        #fm2=pd.DataFrame(fm,index=fm.index)
        ky=fm.keys()
        if 'turnover' in ky:
            del fm['turnover']
        if 'ma5' in ky:
            pp=fm.pop('ma5')
            fm.insert(2,'ma5', pp)
        else:
            ma5=pd.rolling_mean(fm['close'], 5)
            fm.insert(2,'ma5', ma5)
            '''
            fo = open(pathName+stock_id+".csv", "r")
            str1=str(fo.read(-1))
            s = StringIO.StringIO()
            s.write(str1)
            s.seek(0)        
            rws=fm.shape[0]
            v2=pd.read_csv(s, skip_blank_lines=True,sep='\t')
            '''
            
            fm=fm.iloc[5:]
            fm=fm.sort_index( axis=0, ascending=False)
        #fm=fm[:fm.shape[0]-5]
        return fm
        
    def read_hist_data(self, path_name, stock_id, period_id):
        pathName=path_name+period_id+'/'
        
        try:
            fm=pd.read_csv(pathName+stock_id+".csv", '\t')
        except Exception as ex:
            print (ex)
            return None;
        return self.after_read_hist_data( fm)
    
    def write_list_to_csv(self, df, path_name, stock_id, period_id):
        pathName=path_name+period_id+'/max_min_'
        pathName = pathName+stock_id+".csv"
        csvfile = open(pathName, 'wb')
        writer = csv.writer(csvfile)
        writer.writerows(df)
        csvfile.close()
        
    def write_train_input_data(self, dt, name):
        csvfile=litestsl.current_file_directory()+'/'+name
        spamwriter = csv.writer(csvfile)
        for rw in dt:
            spamwriter.writerow(rw)

    def calc_max(self, max_val, i_exclude):
        ls=list(max_val)
        mx=ls[0]
        for i in range(len(ls)):
            if i!=i_exclude and ls[i]>mx:
                mx=ls[i]
        for i in range(len(ls)):
            if i!=i_exclude :
                ls[i]=mx
        return ls
    def calc_min(self, min_val, i_exclude):
        ls=list(min_val)
        mn=ls[0]
        for i in range(len(ls)):
            if i!=i_exclude and ls[i]<mn:
                mn=ls[i]
        for i in range(len(ls)):
            if i!=i_exclude :
                ls[i]=mn
        return ls


    def data_normalization(self, df, stock_id, period_id):
        #df.applymap(lambda x: float(x))
        df['volume']=df['volume'].astype(float)
        if self.normalization_mode==2:
            return df
        df_temp = df
        max_val = list(df_temp.max(axis=0))
        max_val=self.calc_max(max_val, self.max_data_train-1)
        min_val = list(df_temp.min(axis=0))
        min_val=self.calc_min(min_val, self.max_data_train-1)
        mean_val = []#list(((max_val)+(min_val)) / 2)
        for i in range(len(max_val)):
            mean_val.append((min_val[i]+max_val[i])/2)
        nan_values = df_temp.isnull().values
        row_num = len(list(df_temp.values))
        col_num = len(list(df_temp.values)[1])
        dt=[]
        dt.append(max_val)
        dt.append(min_val)
        for rn in range(row_num):
            #data_values_r = list(data_values[rn])
            nan_values_r = list(nan_values[rn])
            for cn in range(col_num):
                if nan_values_r[cn] == False:
                    #v11=df_temp.values[rn][cn] - mean_val[cn]
                    #v22=max_val[cn] - min_val[cn]
                    df_temp.iat[rn,cn] = (df_temp.values[rn][cn])/float(max_val[cn] )
                    #df_temp.iat[rn,cn] = 2 * (v11)/(v22)
                else:
                    print ('Wrong')
        #save to file
        #self.write_list_to_csv(dt, path_name, stock_id, period_id)
         #df_temp.values[rn][cn] = (df_temp.values[rn][cn])/(max_val[cn] )
        #add mean and avariate
        from sklearn import preprocessing
        #df_temp = preprocessing.scale(df_temp,axis = 1)
    
        return df_temp
    
    def read_list_from_csv(self, path_name, stock_id, period_id):
        pathName=path_name+period_id+'/max_min_'
        pathName = pathName+stock_id+".csv"
        csvfile = open(pathName, 'r')
        rd = csv.reader(csvfile)
        rtn=[]
        for row in rd:
            rtn.append(row)
        csvfile.close()
        return rtn
    
    def getStockData( self, stock_id='hs300', period_id=ConstantDefine.period_type['day']):    
        pass
    
    def getStockData2(self, stock_id='399300',  index1=True): 
		pass   
    def getRealtimeData(self, stock_id='399300'):
        df = ts.get_realtime_quotes(stock_id)
        return df
        
        
    def getStockDataToNow(self, stock_id='399300',  index1=True, period_id=ConstantDefine.period_type['day'], data_count_to_now=60):    
        return
    def createARowTrainData(self, i,count_each, n_max, fm3):
        dtRow=[]
        if self.mode_add_ma240:
            f30=fm3['30']
            dtRow.extend(f30.iloc[i:i+2])
            dtRow.extend(fm3['60'].iloc[i:i + 2])
            dtRow.extend(fm3['120'].iloc[i:i + 2])
            dtRow.extend(fm3['240'].iloc[i:i + 2])
        for j in range(count_each):
            v=fm3.iloc[i+j,0:n_max-1]-fm3.iloc[i+j+1,0:n_max-1]
            if not self.mode_use_price:
                v = (v - fm3.iloc[i + j + 1, 0:n_max - 1])/fm3.iloc[i + j + 1, 0:n_max - 1]


            dtRow.extend((v))
        
        dtRow2=[]
        for j in range(count_each):
            v=fm3.iloc[i+j,n_max-1:n_max]
            dtRow2.extend(v)
        
        if self.normalization_mode==2:
                max_price=max(dtRow)
                min_price=min(dtRow)
                mxn=max_price-min_price
                md1=(max_price+min_price)/2.0
                dtRow=list(map(lambda x:(x-md1)/mxn, dtRow))
                max_price=max(dtRow2)
                min_price=min(dtRow2)
                mxn=max_price-min_price
                md1=(max_price+min_price)/2.0
                if mxn==0:
                    dtRow2=list(map(lambda x:0.0, dtRow2))
                else:
                    dtRow2=list(map(lambda x:(x-md1)/mxn, dtRow2))
        # if not self.mode_add_ma240:
        #     dtRowOld = dtRow;
        #     dtRow = []
        #     for j in range(count_each):
        #         dtRow.extend(dtRowOld[j*(n_max-1):(j+1)*(n_max-1)])
        #         dtRow.append(dtRow2[j])
        #
        # else:
        ((dtRow)).extend((dtRow2))
        return dtRow

    def getZfText(self, txt):
        if self.mode_target == 0:
            return txt
        else:
            if self.count_each_vl!=15:
                str1 = "_"+str(self.count_each_vl)
            else:
                str1 = ""
            return txt+str(self.mode_target)+"_"+str(self.zf_max_rate)+str1
    def ma_compute(self,fm3,cunt,kys):
        if str(cunt) not in kys:
            mavl = fm3['close'].rolling(window=cunt, center=False).mean()
            mavl = mavl.shift(-cunt+1)
            fm3[str(cunt)] = mavl
    def createTrainDataNew2(self, fm, ignore_head_rows=False):
        fm3 = fm.copy();
        #self.createTrainDataNextPeriod(fm)

        kys = fm3.keys()
        ma5added = False
        if 'ma5' not in kys:
            ma5 = fm3['close'].rolling(window=5, center=False).mean()
            ma5 = ma5.shift(-4)
            fm3.insert(2, 'ma5', ma5)
            ma5added = True
        #if 'ma10' not in kys:
        ma10 = fm3['close'].rolling(window=10, center=False).mean()
        ma10 = ma10.shift(-9)
        fm3['open'] = ma10
        #ma5added = True

        if self.mode_add_ma240:
            self.ma_compute(fm3, 30, kys)
            lst = 60
            self.ma_compute(fm3,60,kys)
            lst=120
            self.ma_compute(fm3,120,kys)
            lst = 240
            self.ma_compute(fm3,240,kys)
            fm3 = fm3.iloc[:fm3.shape[0] - lst]
        else:
            if ma5added:
                fm3 = fm3.iloc[:fm3.shape[0] - 5]

        if self.mode_target == 0:
            count_each = self.count_each_vl
            n_max = self.max_data_train
            fm2 = []  # pd.DataFrame()

            fm3.applymap(lambda x: float(x))
            n = fm3.shape[0]
            # fm3=fm3.to_array()
            cls = []

            labels = []

            k = 0
            for i in range(n - count_each-1):
                cls_tmp = fm3['close'].iloc[i]
                # dif_p=(cls_tmp-fm3.iloc[i,n_max])/cls_tmp
                # if dif_p<=self.m_max_need and dif_p>=-self.m_max_need:
                #    continue
                dtRow = self.createARowTrainData(i, count_each, n_max, fm3)

                lb = []

                k += 1
                if (not ignore_head_rows) or ignore_head_rows:
                    b_valid = True
                    for j in range(5):
                        if (ignore_head_rows and k <= self.labels_count):
                            if k <= j + 1:
                                break

                        m = 1.0

                        if cls_tmp == fm3.iloc[i, n_max + j]:
                            b_valid = False
                            # break

                        if cls_tmp >= fm3.iloc[i, n_max + j]:
                            m = -1.0
                        lb.append(m)

                labels.append(lb)
                cls.append(cls_tmp)
                fm2.append(dtRow)

        else:
            pass
        return fm2, labels,cls

    def createTrainData2(self, fm, ignore_head_rows=False):
        fm3 = fm.copy();
        #self.createTrainDataNextPeriod(fm)

        kys = fm3.keys()
        ma5added = False
        if 'ma5' not in kys:
            ma5 = fm3['close'].rolling(window=5, center=False).mean()
            ma5 = ma5.shift(-4)
            fm3.insert(2, 'ma5', ma5)
            ma5added = True

        if self.mode_add_ma240:
            self.ma_compute(fm3, 30, kys)
            lst = 60
            self.ma_compute(fm3,60,kys)
            lst=120
            self.ma_compute(fm3,120,kys)
            lst = 240
            self.ma_compute(fm3,240,kys)
            fm3 = fm3.iloc[:fm3.shape[0] - lst]
        else:
            if ma5added:
                fm3 = fm3.iloc[:fm3.shape[0] - 5]

        if self.mode_target == 0:
            count_each = self.count_each_vl
            n_max = self.max_data_train
            fm2 = []  # pd.DataFrame()
            # fm=pd.DataFrame()
            # if 'ma5' not in kys:
            #     ma5 = pd.rolling_mean(fm3['close'], 5)
            #     ma5=ma5.shift(-4)
            #     fm3.insert(2, 'ma5', ma5)
            #     fm3 = fm3.iloc[:fm3.shape[0]-5]

            fm3.applymap(lambda x: float(x))
            n = fm3.shape[0]
            # fm3=fm3.to_array()
            cls = []

            labels = []

            k = 0
            for i in range(n - count_each-1):
                cls_tmp = fm3['close'].iloc[i]
                # dif_p=(cls_tmp-fm3.iloc[i,n_max])/cls_tmp
                # if dif_p<=self.m_max_need and dif_p>=-self.m_max_need:
                #    continue
                dtRow = self.createARowTrainData(i, count_each, n_max, fm3)

                lb = []

                k += 1
                if (not ignore_head_rows) or ignore_head_rows:
                    b_valid = True
                    for j in range(5):
                        if (ignore_head_rows and k <= self.labels_count):
                            if k <= j + 1:
                                break

                        m = 1.0

                        if cls_tmp == fm3.iloc[i, n_max + j]:
                            b_valid = False
                            # break

                        if cls_tmp >= fm3.iloc[i, n_max + j]:
                            m = -1.0
                        lb.append(m)

                labels.append(lb)
                cls.append(cls_tmp)
                fm2.append(dtRow)

        else:
            k=0
            k+=1
        return fm2, labels,cls
    def createTrainDataNextPeriod(self, fm):
        fm3 = fm.copy();
        kys = fm3.keys()
        ma5added = False
        if 'ma5' not in kys:
            ma5 = fm3['close'].rolling(window=5, center=False).mean()
            ma5 = ma5.shift(-4)
            fm3.insert(2, 'ma5', ma5)
            ma5added = True

        if self.mode_add_ma240:
            self.ma_compute(fm3, 30, kys)
            lst = 60
            self.ma_compute(fm3,60,kys)
            lst=120
            self.ma_compute(fm3,120,kys)
            lst = 240
            self.ma_compute(fm3,240,kys)
            fm3 = fm3.iloc[:fm3.shape[0] - lst]
        else:
            if ma5added:
                fm3 = fm3.iloc[:fm3.shape[0] - 5]

        count_each = self.count_each_vl
        n_max = self.max_data_train
        fm2 = []

        if 'ma5' not in kys:
            ma5 = fm3['close'].rolling(window=5, center=False).mean()
            ma5 = ma5.shift(-4)
            fm3.insert(2, 'ma5', ma5)
            fm3 = fm3.iloc[:fm3.shape[0] - 5]

        fm3_old = fm3
        zf5_old = abs(fm3['close'].diff(-1)).rolling(window=5, center=False).mean()
        zf5_old = zf5_old.iloc[4]
        i=0
        for iNext in range(20):
            fm3 = fm3_old[:count_each+5]
            fm3 = fm3.shift(1)
            fm3['close'].iloc[0] = fm3_old['close'].iloc[0] + (iNext - 5) * zf5_old
            fm3['volume'].iloc[0] = fm3['volume'].iloc[1]
            fm3['open'].iloc[0] = fm3['close'].iloc[1]
            fm3['ma5'].iloc[0] = (fm3['close'].iloc[0] + fm3['close'].iloc[1]
                                  + fm3['close'].iloc[2] + fm3['close'].iloc[3] + fm3['close'].iloc[4]) / 5.0

            fm3.applymap(lambda x: float(x))

            dtRow = self.createARowTrainData(i, count_each, n_max, fm3)
            fm2.append(dtRow)
        return fm2, zf5_old

    def createPredictData(self, fm):
        return self.createTrainData2( fm, ignore_head_rows=True)

    def to_float(self, x):
        return float(x)

    def load_trained_data( self, stock_id, period_id, which_data):
        return litestsl.load_trained_data1(stock_id, which_data,'wk')

    def period_id_to_sina_mode(self, period_id):
        rtn=None
        if period_id==ConstantDefine.period_type['day']:
            rtn="daily"
        if period_id==ConstantDefine.period_type['week']:
            rtn="weekly"
        if period_id==ConstantDefine.period_type['month']:
            rtn="monthly"
        return rtn
    
    def code_to_sina_code(self, code_id):
        rtn=None
        
        if code_id=='sh' or code_id=='999999':
            rtn="sh000001"
        if code_id=='sz':
            rtn="sz399001"
        if code_id=='zxb':
            rtn="sz399005"
        if code_id=='cyb':
            rtn="sz399006"
        if rtn is None:    
            if code_id.startswith('6'):
                rtn="sh"+code_id
            if code_id.startswith('00') or code_id.startswith('399'):
                rtn="sz"+code_id
        return rtn

