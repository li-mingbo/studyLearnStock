#coding:utf-8
'''
Created on 2016-1-13

@author: li
'''
import tushare as ts
from numpy import *
import pandas as pd
import os 
import csv

import stockDataHandleWrap as util
import copy

#import StudyApi as studyapi
import litestcy as litest
import sys, getopt
import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.sstyle.use('ggplot')
import cPickle as pk


def createData3(handleWrap, stock_id='600196_1',period_id='D'):
    if handleWrap is None:
        return
    vl=handleWrap.read_hist_data(os.getcwd()+'/', stock_id, period_id)
    vl= handleWrap.transformDataAll(vl,  stock_id, period_id)
    dataArr, labelArr,cls2=handleWrap.createTrainData2(vl)
   
    return dataArr, labelArr,cls2


    
def dump_obj( dt,name):
    f = file('/home/li/'+name+'.lst', 'w')
    pk.dump(dt, f)
    f.close()
def load_obj( name):
    f = file('/home/li/'+name+'.lst', 'r')
    f.seek(0)
    dt=pk.load(f)
    f.close()
    return dt


    
def merge_list(lk, dataArr):
    dataArr2 = copy.deepcopy(dataArr)
    for i in range(len(lk)):
        dataArr2[i].append(lk[i]/10.0)
    return dataArr2

def test2_3(stock_id_arr, period_id='D', id_predict='999999_1'):
    '''
    进行训练学习和认证
    :param stock_id_arr: 股票代码列表， 根据它从 D/ 下装载数据
    :param period_id: 周期字符串
    :return: None
    '''
    #if handleWrap is None:
    handleWrap=util.HandleWrapClass(None)
    handleWrap.mode_target = 0

    handleWrap.count_each_vl=15

    #handleWrap.max_data_train =4
    md_1=0
    stock_id=stock_id_arr[0]
    md_2 = 0

    name_predict = id_predict
    s_w = str(handleWrap.mode_target) + '_' + str(handleWrap.count_each_vl) + '_' + name_predict + '.wpt'

    if md_2==0:
        if True:
            dataArr2, classLabels21,cls2=createData3(handleWrap,name_predict)#JML8_15#IF1603_5min_2    JML8_15
        else:
            sid1 = "RB0"
            import futureHistData as futHD
            vl = futHD.get_hist_data(sid1)
            vl = handleWrap.transformDataAll(vl, sid1, period_id)
            dataArr2, classLabels21, cls2 = handleWrap.createTrainData2(vl)
        dump_obj(dataArr2, "dataArr2")
        dump_obj(classLabels21, "classLabels21")
        dump_obj(cls2, "cls2")
    else:
        dataArr2=load_obj( "dataArr2")
        classLabels21=load_obj( "classLabels21")
        cls2 = load_obj("cls2")
    #litestTensorFlow.train(5, 60, dataArr2, classLabels21, dataArr2, classLabels21)

    dataArr=[]
    classLabels=[]
    if md_1==0:
        #'hs300'
        for stock_id2 in stock_id_arr:
            vl=handleWrap.read_hist_data(os.getcwd()+'/', stock_id2, period_id)
            str1=str(vl)

            vl= handleWrap.transformDataAll(vl,  stock_id2, period_id)
            dataArr3, classLabels3, cls3=handleWrap.createTrainData2(vl)
            dataArr.extend(dataArr3)
            classLabels.extend(classLabels3)
        if False:
            lk = litest.learnKmeans(dataArr,100,50)
            rtn1=litest.classify(dataArr, lk, 0.0001)
            dct1 = {}
            kk=0
            for itm1 in rtn1:
                vkk = classLabels[kk]
                issuc = 0
                for t2 in vkk:
                    if t2 > 0.5:
                        issuc = 1
                        break
                ech = {}
                if dct1.has_key(str(itm1[0])):
                    ech = dct1[str(itm1[0])]
                    ech['all']+=1
                    ech['suc'] += issuc
                else:
                    ech['all'] = 1
                    ech['suc'] = issuc
                    dct1[str(itm1[0])] = ech
                kk+=1
            kk=0
            for (itm1, v1) in dct1.iteritems():
                f1 = float(v1['suc']) / v1['all']
                if f1<0.5:
                    f1 = 1-f1
                v1['rt'] = f1
                kk+=1
            itms = sorted(dct1.items(), lambda x, y: cmp(x[1]['rt'], y[1]['rt']), reverse=True)
            #itms = dct1.items()
            n3 = len(itms)
            n3 = int(n3*0.5)
            itms = itms[n3:]
            dct1 = dict(itms)
            keys = dct1.keys()
            dataArr_tm = []
            classLabels_tm = []
            for (dt, lb, rn) in zip(dataArr, classLabels, rtn1):
                vl = rn[0]
                if dct1.has_key(str(vl)):
                    dataArr_tm.append(dt)
                    classLabels_tm.append(lb)
            dataArr=dataArr_tm
            classLabels = classLabels_tm
        dump_obj(dataArr, "dataArr")
        dump_obj(classLabels, "classLabels")
    else:
        dataArr=load_obj( "dataArr")
        classLabels=load_obj( "classLabels")
        #cls2 = load_obj("cls2")

    if False:
        dataArr.extend(load_obj("data_"+s_w))
        classLabels.extend(load_obj("labl_"+s_w))
    #litest2.train(5, 60, dataArr, classLabels, dataArr2, classLabels21)
    #import litestRnnBi
    # import litestTensflowOld
    # litestTensflowOld.train(2, 60, dataArr, classLabels, dataArr2, classLabels21)
    # return

    result1=pd.DataFrame()
    res_sum=None
    cls=False
    result2=[]
    result3=pd.DataFrame()
    result1['sum1']=[]
    

    for j in range(5):
        if md_1==0:
            lb=mat(classLabels)
            classLabels5=lb[:,j]
            classLabels5=array(classLabels5.T)
            # lk=litest.learnKmeans(dataArr)
            # litest.dump_trained_data1( lk, stock_id, j, 'lk')
            # lb = map(lambda xx: float64(xx), lk.labels_)
            # dataArrTmp=merge_list(lb, dataArr)#dataArr#
            wk,classArr=litest.train(dataArr, classLabels5[0],50)
            litest.dump_trained_data1( wk, stock_id, j, handleWrap.getZfText('wk'))
            #dump_trained_data1( classArr, stock_id, period_id, j, 'classArr')
        
        wk2=litest.load_trained_data1(  stock_id, j, handleWrap.getZfText('wk'))
        #classArr_2=load_trained_data1(  stock_id, period_id, j, 'classArr')
        wk=wk2
         
        #handleWrap.dump_trained_data( wk, stock_id, period_id, j)
        #wk=handleWrap.load_trained_data(stock_id, period_id, j)
        lb=mat(classLabels21)
        classLabels2=lb[:,j]
        classLabels2=array(classLabels2)

        k1=len(classLabels2)
        tmp1=mat(ones((k1,1)))
        tmp1=tmp1.T
        if cls==False:
            cls=True
            mx1=min(cls2)
            mx3=max(cls2)
            mx1=float(mx1)
            cls2=map(lambda x: ((x-mx1)/(mx3-mx1)*(12))*2,cls2)
            result1['close']=cls2
            
        # lk=litest.load_trained_data1(  stock_id, j, 'lk')
        # lk_p=lk.predict(dataArr2)
        # lb2 = map(lambda xx: float64(xx), lk_p)

        #dataArrTmp2=merge_list(lb2, dataArr2)#dataArr2#
        pr=litest.classify(dataArr2, wk,0.0001)
        pr=array(mat(pr))
        if j==0:
            result1[str(j+1)]=pr#array(mat(pr))
            result1['0']=where(pr==classLabels2,-2,-4)#array(mat(pr))
        else:
            result1[str(j+1)]=array(mat(pr)+(tmp1.T)*j)
        result3[str(j)]=array(mat(pr).T)[0]
        
        if res_sum is None:
            res_sum=pr
        else:
            res_sum=array(mat(pr)+mat(res_sum))
        if j==4:
            res_sum=array(mat(res_sum)+mat(ones((k1,1)))*12)
        
        err=mat(ones((k1,1)))

        mt_label=mat(classLabels2)
        mt_label=where(pr==0,0,mt_label)
        #k=err[pr.any()!=mt_label.any() and mt_label.any()!=0].sum()
        k=0#err[pr!=mt_label].sum()
        k5=0
        k6=0
        k7=0
        k8=0
        
        for k3 in range(len(mt_label)):
            if mt_label[k3]==0 and pr[k3]==0:
                k5+=1
            if mt_label[k3] != pr[k3]:
                k6+=1
            if mt_label[k3] == 1 and  pr[k3] == 1:
                k8+=1
            if pr[k3]>0:
                k7+=1
            if mt_label[k3] == pr[k3]:
                k+=1

        #print pr
        
        k-=k5
        print ("K:"+str(k)+", K1:"+str(k1)+", K6:"+str(k6)+", K7:"+str(k7)+", K8:"+str(k8))
        print k/float(k1-k5)
        result2.append(k/float(k1-k5))
    result1['sum1']=res_sum
    '''to compute other '''
    litest.output_result3(result3, classLabels21, 1)
    litest.output_result3(result3, classLabels21, 0)
    ij=0
    tmp2 = []

    if handleWrap.mode_target>0:
        w_data = []
        w_label = []
        ln1=len(classLabels21)
        mtx = result3.as_matrix()
        for it in classLabels21:
            n=0
            n2=0
            for m in range(5):
                if it[m]>0.5:
                    n+=1
                if mtx[ij,m]>0.5:
                    n2 += 1
            if n2>0 and n>0:
                w_data.append(dataArr2[ij])
                w_label.append(it)
                tmp2.append(1-8)
            elif n2>0:
                tmp2.append(1 - 9)
            else:
                tmp2.append(1 - 10)
            ij+=1
    #     if len(w_data)>0:
    #         dump_obj(w_data, "data_"+s_w)
    #         dump_obj(w_label, "labl_"+s_w)

        result1['a'] = tmp2

    result1=result1.sort_index( axis=0, ascending=False)
    plt.figure();result1.plot()#
    k=0
    #mo hu status mothod



def usage_help():
    print '-t is type of operation, 0 is download stock data; 1 is training;2 is predict'
    print '-h is this help'
    print '-s is stock code, for example: 399300, 600028'
    print '-i is the stock whether is index, 0 is not index, 1 is index'
    print '-x is the stock code of knowledge when predict'
    print '-p is the period of stock data, D is day'

    
if __name__ == '__main__':
    '''
        predicts()999999_2
    #999999_5 is very good999999_2
    arr=['999999_2_all','000003_2','399005_2','399006_2','399300_2']
    '''
    arr = ['999999_2']
    arr = ['399300_2']

    #use this function to train and analyze predicting result
    test2_3(arr,'D','399300_1')

'''
'''    