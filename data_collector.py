import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries
import collections
import csv
import time

import logger


np.set_printoptions(suppress=True)

class Collector():
    '''
    class to extract data from alpha vantage api
    default storage to proj home /store
    '''
    def __init__(self, control_file=None, key=None):
	self.log = logger.Logging()
	self.control_file = control_file
	self.key = key
	if control_file != None:
	    #PLEASE use control file and config dict
	    self.config = self.read_control_file()
	    #remember the value in the config dict is a list so index with [0], etc
	    self.symbols = self.config['stocks'][0].split(',')
	    self.key = self.config['api_key'][0]
	else:
	    self.config = {}
	    self.symbols = []
	try:
	    self.ts = TimeSeries(key=self.key, output_format='pandas')
	except:
	    self.ts = None
	    self.log.error('TimeSeries failed, check key')
	#setup storage for instance usage, key=symbol, val=[data]
	self.storage = collections.defaultdict(list)
	self.symbol = pd.DataFrame()
	
    def get_daily(self, sz='compact', sym=None):
	'''
	wrapper for get_daily_adjusted - use control file!! if not, single symbol accepted
	'''
	if sym != None:
	    #in case we want to do this on the fly for one symbol
	    try:
	        data, meta_data = self.ts.get_daily_adjusted(sym, outputsize=sz)
	        self.symbol = data
		self.log.info('collector fetched data for {}'.format(sym))
	    except:
		self.log.error('could not fetch data for {}'.format(ticker))
	else:
	    #loop through all config symbols and call fetch, add to list - size is +1MB each
	    for ticker in self.symbols:
		try:
	            self.log.info('fetching {}'.format(ticker))
	    	    data, meta_data = self.ts.get_daily_adjusted(ticker, outputsize=sz)
		    #rename columns so it isnt ridiculous
		    data = self.rename_cols(data)
		    self.storage[ticker] = data
		except Exception as e:
		    self.log.error('could not fetch data for {}\n{}'.format(ticker, e))
	    #make the api call level go down
	        time.sleep(5)
	    self.log.info('collector fetched data for {} symbols'.format(len(self.storage)))
	return 0

    def read_control_file(self, filename=None):
	'''
	initilize object with a control file versus parameters or cli input
	input: filename, file in format
	KEY:VALUE
	or
	KEY:VALUE,VALUE,...
	'''
	if filename==None:
	    filename = self.control_file
	try:
	    control = {}
	    with open(filename, 'r') as f:
		reader = csv.reader(f, delimiter=':')
		for row in reader:
		    control[row[0]] = row[1:]
	    return control
	except:
	    self.log.error('failed to process control file')

    def rename_cols(self, df, cols=None):
	'''
	rename alpha columns for canonical symbol gets - redo this for various other data pulls
	need to change column order still
	'''
	if cols == None:
	    cols = ['low', 'open', 'high', 'close', 'volume', 'adj_close', 'split_coeff', 'dividend']
	df.columns = cols
	return df

    def save_storage(self, postfix='100d', dr='store/', sym_list=None):
	'''
	write data frames out to store/ directory
	must supply directory as dr, postfix as number and d=days, m=minutes
	'''
	if sym_list == None:
	    for sym, data in self.storage.items():
		#np.savetxt('{}{}_{}.csv'.format(dr,sym,postfix), data.values, delimiter=',')
		data.to_csv('{}{}_{}.csv'.format(dr,sym,postfix), sep=',')
		self.log.info('saved {} to store/'.format(sym))
	else:
	    return -1

    def load_storage(self, sym):
	#implement
	return 0

if __name__ == '__main__':
    collector = Collector('config.txt')
    print collector.config
    #print collector.key
    print collector.symbols
    collector.get_daily(sz='full')
    #print collector.storage['nvda']
    collector.save_storage(postfix='5953d')

