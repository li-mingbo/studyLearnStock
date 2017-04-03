'''
Created on Dec 15, 2016
li is short for machine study
@author: li mingbo
'''
from numpy import *
import datetime
import os 
import sys
import json

# function: get directory of current script, if script is built
#   into an executable file, get directory of the excutable file
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

from sklearn.cluster import KMeans

use_forest = True

def current_file_directory():
            import inspect
            path = os.path.realpath(sys.path[0])        # interpreter starter's path
            if os.path.isfile(path):                    # starter is excutable file
                path = os.path.dirname(path)
                return os.path.abspath(path)            # return excutable file's directory
            else:                                       # starter is python script
                caller_file = inspect.stack()[1][1]     # function caller's filename
                return os.path.abspath(os.path.dirname(caller_file))# return function caller's file's directory

def getPath(stock_id, period_id, which_data, name):
    return current_file_directory()+'/'+period_id+ '/'+str( which_data)+'_trained_'+stock_id+'_'+name+'.trn'

def is_trained_data_existing( stock_id, which_data, name):
    period_id2="trained"
    b=os.path.exists(getPath(stock_id, period_id2, which_data, name))
    return b
def to_percent(f_val):
    val=str(((int)(f_val*10000))/100.0)+"%"
    return val
def out_success_rate_result(result3, classLabels21, rtn_dict=None):
    if rtn_dict is None:
        return
    mtx=None
    result_0=None
    sm=None
    n1=0
    n2=0
    n_dif=0
    n_dif2=0
    
    try:
        mtx=result3.as_matrix()
        result_0=where(mtx==0,1,0)
        result_0_1=result_0[1:,0]
        result_0_2=result_0[2:,1]
        n1=int(result_0_1.sum(0))
        n2=int(result_0_2.sum(0))
        #mtx=where(result_0==1,0,mtx)
        mtx1=(classLabels21)
        mtx1=(mtx1[1:])
        len1=len(mtx1)
        mtx11=[mtx1[i][0] for i in range(len1)]#mtx1[:][0]
        mtx1=(mtx1[1:])
        len1-=1
        mtx12=[mtx1[i][1] for i in range(len1)]
        mtx11=where(result_0_1==1,0,mtx11)
        mtx12=where(result_0_2==1,0,mtx12)
        mtx11=mat(mtx11)
        mtx11=mtx11.T
        mtx12=mat(mtx12)
        mtx12=mtx12.T
        len1=len(mtx11)
        rtn_dict['detail_src_list']=[mtx11[i,0] for i in range( len1 )]#list(array(mtx11))
        
        mtx=mat(mtx)
        mtx01=mtx[1:,0]
        en1=len(mtx01)
        rtn_dict['detail_list']=[mtx01[i,0] for i in range( len1 )]#list(array(mtx01))
        
        mtx01=where(mtx11!=mtx01,1,0)
        n_dif=int(mtx01.sum(0))
               
        mtx02=mtx[2:,1]
        mtx02=where(mtx12!=mtx02,1,0)
        n_dif2=int(mtx02.sum(0))
        k1=len(classLabels21)-1
        rtn_dict["success_num"]=[int(k1-n_dif-n1),int( k1-n_dif2-n2-1)]
        rtn_dict["fail_num"]=[int(n_dif),int(n_dif2)]
        rtn_dict["unknown_num"]=[n1,n2]
    except Exception as ex:
        print (ex)
        
    
def output_result3(result3, classLabels21, mode_up, rtn_dict=None):
    mtx=None
    mtx2=None
    result_0=None
    sm=None
    sm2=None
    sm3=None
    sm31=None
    nn=0
    nn2=0
    n_dif=0
    
    try:
        mtx=result3.as_matrix()
        result_0=where(mtx==0,1,0)
        nn2=result_0[:,0].sum(0)
        #mtx=where(result_0==1,0,mtx)
        mtx=mat(mtx)
        if mode_up==1:
            mtx2=where(mtx>0,1,0)
        else:
            mtx2=where(mtx<0,1,0)
    except Exception as ex:
        print ('1111')
        print (ex)
    try:
    
        mtx2=mat(mtx2)
        sm=mtx2.sum(1)
        sm=mat(sm)
        
        nn=result_0.sum(1)
        nn=where(nn==5,1,0)
        nn=nn.sum(0)
        
        #nn=[[0]]
    except Exception as ex:
        print ('333')
        print (ex)
    try:   
        mtx1=mat(classLabels21)
        mtx1=where(result_0==1,0,mtx1)
        
        if mode_up==1:
            mtx2=where(mtx1>0,1,0)
        else:
            mtx2=where(mtx1<0,1,0)

        mtx2=mat(mtx2)
        
        sm2=mat(mtx2.sum(1))

        sm3=2*sm2-sm
        sm31=sm2-sm
    except Exception as ex:
        print ('555')
        print (ex)
    try:   

    
        sm3=where(sm3>=0 ,1,0)
        sm31=where(sm31>=0 ,1,0)
        sm3_0=where(sm2==0 ,1,0)
        val_0 = sm3_0.sum(0)
        sm3_0=where(sm==0 ,sm3_0,0)

        val_0 = sm3_0.sum(0)

        sm5 = where(sm>0.5 ,1,0)
        #sm5 = where(sm<1.5 ,sm5,0)
        sm5_out = where(sm2 > 0.5, sm5, 0)
        sm5 = where(sm5 > 0.5, 1, 0)
        sm5_out = where(sm5_out > 0.5, 1, 0)

        val=sm3.sum(0)-nn - val_0
        val2=sm31.sum(0)-nn - val_0
        k1=len(classLabels21)-nn-val_0
        #k1-=val5
        s='val_down: '
        if mode_up==1:
            s='val_up: '
        #strall=to_percent(val2.tolist()[0][0]/float(k1))
        #strhalf=to_percent(val.tolist()[0][0]/float(k1))
        strall=to_percent(val2/float(k1))
        strhalf=to_percent(val/float(k1))
        print (s+strall+'  half: '+strhalf +" mode2: "+to_percent(sm5_out.sum(0)/float(sm5.sum(0))))
        
        if rtn_dict is None:
            return
        rtn_dict["count"]=k1
        if not rtn_dict is None:
            if mode_up==1:
                rtn_dict["high_num"]=int(sm[0])
                rtn_dict["high"]=strall
                rtn_dict["high_half"]=strhalf
            else:
                rtn_dict["low_num"]=int(sm[0])
                rtn_dict["low"]=strall
                rtn_dict["low_half"]=strhalf
    except Exception as ex:
        print ('666')
        print (ex)

def output_result(result3, classLabels21, mode_up, rtn_dict):
    mtx=result3.as_matrix()
    mtx=mat(mtx)
    if mode_up==1:
        mtx2=where(mtx>0,1,0)
    else:
        mtx2=where(mtx<0,1,0)
    mtx2=mat(mtx2)
    
    sm=(mtx2.sum(1))
    sm=mat(sm)
    mtx=mat(classLabels21)
    if mode_up==1:
        mtx2=where(mtx>0,1,0)
    else:
        mtx2=where(mtx<0,1,0)
    mtx2=mat(mtx2)
    
    sm2=mat(mtx2.sum(1))
    sm3=sm2-(sm)/2
    sm31=sm2-(sm)
    sm3=where(sm3>0,1,0)
    sm31=where(sm31>0,1,0)
    sm3=mat(sm3)
    sm31=mat(sm31)
    
    val=sm3.sum(0)
    val2=sm31.sum(0)
    k1=len(classLabels21)
    s='val_down: '
    if mode_up==1:
        s='val_up: '
    strall=to_percent(1-val2.tolist()[0][0]/float(k1))
    strhalf=to_percent(1-val.tolist()[0][0]/float(k1))
    print (s+strall+'  half: '+strhalf )
    
    rtn_dict["count"]=k1
    if not rtn_dict is None:
        if mode_up==1:
            rtn_dict["high_num"]=sm[0]
            rtn_dict["high"]=strall
            rtn_dict["high_half"]=strhalf
        else:
            rtn_dict["low_num"]=sm[0]
            rtn_dict["low"]=strall
            rtn_dict["low_half"]=strhalf
    #{"high_num":"3","high":"60.1","high_half":"83.8","low_num":"2","low":"61.5","low_half":"81.7"}
    return

import cPickle as pk
def dump_trained_data1( dt, stock_id, which_data, name):
    period_id2="trained"
    path=current_file_directory()+'/'+period_id2
    if not os.path.exists(path):
        os.mkdir(path)
    
    f = open(getPath(stock_id, period_id2, which_data, name), 'w')
    if use_forest:
        pk.dump(dt, f)
    else:
        s=json.dumps(dt, default=lambda dt: dt.__dict__)
        f.write(s)
    #
    f.close()
def load_trained_data1(  stock_id, which_data, name):
    period_id2="trained"
    if not is_trained_data_existing( stock_id, which_data, name):
        return None
    f = open(getPath(stock_id, period_id2, which_data, name), 'r') 
    f.seek(0)

    if use_forest:
        dt = pk.load(f)
    else:
        s = f.read()
        dt=json.loads(s)#, object_hook=string_to_list)

    #dt=pk.load(f)
    f.close()
    return dt


def train(dataArray,classLabels,countIt=50):
    if use_forest:
        aggClass2 = trainRandomForest(dataArray, classLabels)
        return aggClass2,1


def classify(datToClass, classifierArr, limitUnclear=0.001):
     if use_forest:
        aggClass2 = classifierArr.predict(datToClass)
        if not aggClass2 is None:
            aggClass2 = array(mat(aggClass2).T)

        return aggClass2



def trainRandomForest(data, y):
    clf = RandomForestClassifier(n_estimators=50)#61.33
    clf = clf.fit(data, y)
    return clf

def learnKmeans(data, clusters =8, init=10):
    lnk=KMeans(init='k-means++', n_clusters=clusters, n_init=init)
    lnk.fit(data)
    return lnk# lnk.labels_
