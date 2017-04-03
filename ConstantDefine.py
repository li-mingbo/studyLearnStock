#coding:utf-8
'''
Created on 2016-3-29

@author: li
'''
stock_type={'stock':'gp', 'future':'qh', 'stock_index':'gpzs'}
stock_type_name={'gp':u'股票', 'qh':u'期货', 'gpzs':u'股票指数'}
period_type={'5_minute':'5','15_minute':'15','30_minute':'30','60_minute':'60','day':'D','week':'W','month':'M' }
period_type_name2={'5':u'5分钟','15':u'15分钟','30':u'30分钟','60':u'60分钟','D':u'日线','W':u'周线','M':u'月线' }

def create_key(stock_code, code_type=stock_type['stock'], period=period_type['day']):
    return stock_code+'_'+code_type+'_'+period

if __name__ == '__main__':
    pass