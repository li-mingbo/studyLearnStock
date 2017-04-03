#coding:utf-8
from __future__ import division

import time
import json
import lxml.html
from lxml import etree
import pandas as pd
import numpy as np
from tushare.stock import cons as ct
import re
from pandas.compat import StringIO
from tushare.util import dateu as du
try:
    from urllib.request import urlopen, Request
except ImportError:
    from urllib2 import urlopen, Request

K_LABELS = ['D', 'W', 'M']
K_MIN_LABELS = ['1', '5', '15', '30', '60']
K_TYPE = {'D': 'akdaily', 'W': 'akweekly', 'M': 'akmonthly'}
TT_K_TYPE = {'D': 'day', 'W': 'week', 'M': 'month'}

SYMBOLS_FUTURE = {
}
PRICE_COLUMNS = ['date', 'open', 'high', 'low', 'close', 'volume']

def get_hist_data(code=None, start=None, end=None,
                  ktype='D', retry_count=3,
                  pause=0.001):
     raise IOError(ct.NETWORK_URL_ERROR_MSG)
