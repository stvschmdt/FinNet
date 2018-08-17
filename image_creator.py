import pandas as pd

import logger


#class to read in csv from store plus operations
class ImageCreator():
    def __init__(self, filename, n=30):
	self.filename = filename
	self.ndays = n
	self.log = logger.Logging()
	#print 'running for days:', n
	self.log.info('running for days: {}'.format(n))
	self.data = self.load_ohlc_csv()

    def load_ohlc_csv(self):
	try:
	    data = pd.read_csv(self.filename)
	    return data
	except:
	    self.log.error('could not read file {}'.format(self.filename))

    #return either n day window from self.data or pass in dataframe
    def n_day(n, df=None):
	if df == None:
	    return self.data.iloc[:n]
	else:
	    return df.iloc[:n]

    def rolling_window(n, df=None):
#return a rolling window of n days through the dataframe - n days
	return 0

    def create_image_from_np_array(arr):
	return 0
    
    def save_image(im, im_filename):
	return 0

    def get_num_windows(n):
	return 0

    def read_image_to_np_array(im_filename):
	return 0

    #do all the things tested in the 'if' statement below
    def driver(self):
	return 0



if __name__ == '__main__':
    days = [30,60,90]
    for d in days:
        ic = ImageCreator('store/goog_100d.csv', d)
        ic.driver()
    print 'read ',len(ic.data)
    #do step 1, step 2, step 3
    
