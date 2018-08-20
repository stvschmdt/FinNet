import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.finance as fin
import logger
from PIL import Image
from matplotlib.finance import candlestick_ohlc

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

    def n_day(self,n, df=None):
        ### Tie into rolling window function
        #return either n day window from self.data or pass in dataframe
	if df is None:
	    return self.data.iloc[:n]
	else:
	    return df.iloc[:n]

    def convert_dataframe_to_np_array(self,df=None):
        ### returns values of dataframe for numpy array
        if df is None:
            return self.data.values
        else:
            return df.values
        

    def rolling_window(self, df=None):
        ### Rolling window is a generator function that only returns one window at a time
        ### Passes window into create_image_from_np_array function
        ### Window issnapshot of ndays moving forward
        ### Tie this in to nday function
        if df is None:
            num_iter = self.get_num_windows(self.ndays)
            for i in range(num_iter):
                yield self.data.iloc[i:self.ndays+i]
        else:
            num_iter = self.get_num_windows(self.ndays,df=df)
            for i in range(num_iter):
                yield df.iloc[i:self.ndays+i]
    def change_date(self,arr):
        ### Changes date string into index for ndays
        for j in range(len(arr)):
            arr[j][0] = j
        return arr

    def create_image_from_np_array(self,arr):
        #create candlestick chart with matplotlib
        fig = plt.figure()
        ax1 = plt.gca()
        #x limits are the low and high indices, day 0 to day nday for window 1
        # y limits are the min and max of the ohlc data
        plt.xlim(arr[0][0],arr[-1][0])
        print arr[0][-1]
        plt.ylim(np.min(arr[:,1:5]),np.max(arr[:,1:5]))
        #turn off axes around the plot
        plt.axis('off')
        plt.tight_layout()
        candlestick_ohlc(ax1,arr,colorup='#77d879', colordown='#db3f3f')
        ### Saving image here because I'm not sure how to pass to save_image function
        ### Maybe pass candlestick object?
        plt.savefig('./temp_im.png')
#       ax1.set_facecolor('w')
        #plt.Axes(fig,[0,0,1,1])
    
    def save_image(self,im, im_filename):
	return 0

    def get_num_windows(self,n,df=None):
        if df is None:
	    return len(self.data) - n
        else:
            return len(df) - n

    def read_image_to_np_array(self,im_filename):
        ### Reads saved image and converts to a numpy array
        return np.asarray(Image.open(im_filename))

    def delete_alpha(self,arr):
        ### Remove opacity channel from image array
        ### Would keeping this in be more accurate?
        return np.delete(arr,3,2)
        '''
        convert matplotlib figure to 4D numpy array (RGBA channels). Alpha channel (opacity) should be removed.
        No need for writing or reading an image.
        This would save runtime, but the shape of the resulting array is different from pulling a saved array
        with np.asarray, and I am not sure why. This is the same method used in converter.py 

        http://www.icare.univ-lille1.fr/wiki/index.php/How_to_convert_a_matplotlib_figure_to_a_numpy_array_or_a_PIL_image

        
        
        # draw the renderer
        fig.canvas.draw()

        # Get the RGBA Buffer from the figure
        w,h = fig.canvas.get_width_height()
        buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w,h,4)

        #canvas.tostring_argb give pixmap in ARGB mode. Roll the alpha channel to have it in RGBA mode
        buf = np.roll(buf,3,axis = 2)

	return buf
        '''
    def flatten_image(self,arr):
        ### Flatten image to vector for easier saving and appending of label
        return 0

    def append_label(self,arr):
        ### Calculate percentage change +m days after window and append to image
        return 0

    #do all the things tested in the 'if' statement below
    def driver(self):
        ### temporary path for the images to be saved at. There's probably a better way to do this
        temp_path = './temp_im.png'
        generator = ic.rolling_window()
        count = 0
        #generator function only returns one window at a time, potentially speeding image making
        for i in generator:
            self.log.info("rolling window num:{}".format(count))
            count += 1
            arr = self.convert_dataframe_to_np_array(df=i)
            arr = self.change_date(arr)
            self.log.info("saving image...")
            self.create_image_from_np_array(arr)
            self.log.info("reading image...")
            arr = self.read_image_to_np_array(temp_path)
            
            arr = self.delete_alpha(arr)
            self.log.info("saving array...")
	    np.save('./imgs_as_arrays/img_'+str(self.ndays)+'_days_window_'+str(count),arr)
            #Check image after deleting alpha array
            img = Image.fromarray(arr,'RGB')
            img.show()



if __name__ == '__main__':
    check_driver = 1
    days = [30,60,90]

    if check_driver == 1:
        ic = ImageCreator('store/goog_100d.csv', 90)
        ic.driver()
    else:
        ic = ImageCreator('./store/goog_100d.csv')
        # Default is 30 day window
        generator = ic.rolling_window()
        count = 0
        for i in generator:
            count += 1
            arr = ic.convert_dataframe_to_np_array(df=i)
            #Create figure and axes objects, axes needs to be passed to candlestick
            fig = plt.figure()
            ax1= plt.gca()

            #Replace date strings with indices
            for i in range(len(arr)):
                arr[i][0] = i
            #create candlestick chart with matplotlib, hexadecimals create green and red candlesticks
            candlestick_ohlc(ax1,arr,colorup='#77d879', colordown='#db3f3f')
#           ax1.set_facecolor('w')
            plt.tight_layout()
            #turn off axes around the plot
            plt.axis('off')
            #plt.Axes(fig,[0,0,1,1])
            plt.xlim(0,len(arr))
            # y limits are the min and max of the ohlc data
            plt.ylim(np.min(arr[:,1:5]),np.max(arr[:,1:5]))
            plt.savefig('./test1.png')
            plt.show()
            img = np.asarray(Image.open('./test1.png'))
    

    #do step 1, step 2, step 3
    
