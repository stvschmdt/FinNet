import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mpl_finance as fin
import logger
from PIL import Image
from mpl_finance import candlestick_ohlc
import os


#format of API renamed from data_collector.py
#cols = ['low', 'open', 'high', 'close', 'volume', 'adj_close', 'split_coeff', 'dividend']
#class to read in csv from store plus operations
### Add symbol directory to imags_as_arrays, i.e. /imgs_as_arrays/amzn_5953d/30d, or 90d
class ImageCreator():
    def __init__(self, filename, n=30, m = 2):
        self.percent_label_days = m
	self.filename = filename
	self.ndays = n
	self.log = logger.Logging()
	self.log.info('running for days: {}'.format(n))
	self.data = self.load_ohlc_csv()
        self.ohlc = self.reorder_data_columns()
        self.write_dir = self.check_dir()

    def check_dir(self,filename=None,n=None):
        ### Check symbol directory and data directory
        ### if symbol not in directory, create new directory with symbol
        ### Check symbol directory for nday windows
        ### if no nday windows directory, create new with n + 'd'
        if filename == None:
            filename = self.filename
        if n == None:
            n = self.ndays
        ### filename is './store/symbol_numd.csv'
        ### first split yields '.','store','symbol_numd.csv'
        ### second split yields 'symbol_numd'
        ### 'num' + 'd' not to be confused with 'n' + 'd'
        _,symbol = filename.split("/")
        symbol,_ = symbol.split(".")
        n = str(n)+'d'
        ### Switch os.system calls to subprocess calls 
        if symbol in os.listdir('./imgs_as_arrays'):
            if n in os.listdir('./imgs_as_arrays/'+symbol):
                write_dir = './imgs_as_arrays/'+symbol+'/'+n+'/'
            else:
                self.log.info('creating directory ./imgs_as_arrays/'+symbol+'/{}'.format(n))
                os.system('mkdir ./imgs_as_arrays/'+symbol+'/'+n)
                write_dir = './imgs_as_arrays'+symbol+'/'+n+'/'
        else:
            self.log.info('creating directories ./imgs_as_arrays/{}'.format(symbol))
            self.log.info('and ./imgs_as_arrays/'+symbol+'/{}'.format(n))
            os.system('mkdir ./imgs_as_arrays/'+symbol)
            os.system('mkdir ./imgs_as_arrays/'+symbol+'/'+n)
            write_dir = './imgs_as_arrays/'+symbol+'/'+n+'/'
        return write_dir
                

    def load_ohlc_csv(self):
	try:
	    data = pd.read_csv(self.filename)
	    return data
	except:
	    self.log.error('could not read file {}'.format(self.filename))

    def reorder_data_columns(self,data = None):
        try:
            if data is None:
                data = self.data
            ohlc = data[['date','open','high','low','adj_close']]
            return ohlc
        except:
            self.log.error('could not reorder data columns')

    def n_day(self,window_start,window_stop, df=None):
        #return either n day window from self.data or pass in dataframe
	if df is None:
	    return self.ohlc.iloc[window_start:window_stop]
	else:
	    return df.iloc[window_start:window_stop]

    def convert_dataframe_to_np_array(self,df=None):
        ### returns values of dataframe for numpy array
        if df is None:
            return self.ohlc.values
        else:
            return df.values
        

    def rolling_window(self, df=None):
        ### Rolling window is a generator function that only returns one window at a time
        ### Passes window into create_image_from_np_array function
        ### Window is snapshot of ndays moving forward
        if df is None:
            num_iter = self.get_num_windows(self.ndays)
            if num_iter <= 0:
	        self.log.error('price on label day {}'.format(self.ndays+self.percent_label_days))
                self.log.error('is past last day in data {}'.format(len(self.load_ohlc_csv()))) 
            else:
                for i in range(num_iter):
                    yield self.n_day(i,self.ndays+i)
        else:
            num_iter = self.get_num_windows(self.ndays,df=df)
            if num_iter <= 0:
	        self.log.error('price on label day'.format(self.ndays+self.percent_label_days-1))
                self.log.error('is past last day in data'.format(len(load_ohlc_csv(self.filename))+1)) 
            else:
                for i in range(num_iter):
                    yield self.n_day(i,self.ndays+i)
    def change_date(self,arr,count):
        ### Changes date string into index, adds count so that indices will be (for n = 30):
        ### 0-29 for window 0; 1-30 for window 1; 2-31 for window 2; etc
        for j in range(len(arr)):
            arr[j][0] = j - 1 + count
        return arr

    def create_image_from_np_array(self,arr,savepath, n=None):
        if n == None:
            n = self.ndays
        #create candlestick chart with matplotlib
        fig = plt.figure()
        ax1 = plt.gca()
        #x limits are the low and high indices, day 0 to day nday for window 1
        # y limits are the min and max of the ohlc data
        plt.xlim((0,n))
        plt.ylim(np.min(arr[:,1:5]),np.max(arr[:,1:5]))
        #turn off axes around the plot
        plt.axis('off')
        plt.tight_layout()
        candlestick_ohlc(ax1,arr,colorup='#77d879', colordown='#db3f3f')
        ### Saving image here because I'm not sure how to pass to save_image function
        ### Maybe pass candlestick object?
        plt.savefig(savepath+'.png',dpi = 200)
#       ax1.set_facecolor('w')
        #plt.Axes(fig,[0,0,1,1])
    
    def save_image(self,im, im_filename):
	return 0

    def get_num_windows(self,n,df=None):
        ### When appending label, cannot put label for windows where index is out of range
        ###if n = 90, cannot append 2 days out label for window 9-98 because index 100 is
        ### out of range for data frame, so we subtract self.percent_label_days and add 1 to number of windows
        if df is None:
	    return len(self.ohlc) - n - self.percent_label_days + 1
        else:
            return len(df) - n - self.percent_label_days + 1

    def read_image_to_np_array(self,im_filename):
        ### Reads saved image and converts to a numpy array
        return np.asarray(Image.open(im_filename+'.png'))

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
	### 'C' parameter flattens C style, along rows
	### mapped as [i,j,k] -> [j + rowlen*i + k*rowlen*collen]
        ### DONT USE ARR1, TO SAVE SPACE
	arr = arr.flatten('C')
        return arr

    def get_labels(self,arr,img,df = None):
        ### Calculate percentage change +m days after window
        ### Note that the entire data frame must be passed to df rather than a windowed data frame
        ### This is because the windowed frame doesn't contain the future price
        ### Will need shape in order to recreate image array
        ### numpy arrays do not preserve the images when appending and deleting. so we need a list
        if df is None:
            dataline = self.ohlc.iloc[arr[-1][0] + self.percent_label_days]
            percent_diff = (np.max(dataline[1:4]) - self.ohlc.iloc[-1][4])/np.max(dataline[1:4])
            return [img.shape[0],img.shape[1],img.shape[2],float(100*percent_diff)]
        else:
            dataline = df.iloc[arr[-1][0] + self.percent_label_days]
            percent_diff = (np.max(dataline[1:4]) - df.iloc[-1][4])/np.max(dataline[1:4])
            return [img.shape[0],img.shape[1],img.shape[2],float(100*percent_diff)]

    def append_labels(self,arr,labels):
        ### Add label and shape to the end of flattened image array
        arr = list(arr)
        print arr[10:20]
        for i in labels:
            arr.append(i)
        print arr[10:20]
        return arr

    def save_array(self,img_arr,count):
        write_file = self.write_dir + 'window' + str(count) + '_label'+str(self.percent_label_days)+'d'
        with open(write_file,'w') as f:
            for num in img_arr:
                f.write("%s\n" % num)
        return write_file 

    def recreate_image(self,write_file):
        ### Recreate image from numpy file
        ### This doesn't work yet. The recreated images are corrupted somehow
        arr = []
        with open(write_file) as f:
            arr = [line.rstrip('\n') for line in f]
        img_arr = map(int,arr[:len(arr)-4])
        img_arr = np.asarray(img_arr).astype('uint8')

        ### peel off labels and remove from image array
        percent_label = float(arr[-1])
        shape = (int(arr[-4]),int(arr[-3]),int(arr[-2])) 
        print shape
        #print percent_label, shape0, shape1, shape2
        ### construct image array
        img_arr = np.reshape(img_arr,shape)
        image = Image.fromarray(img_arr,'RGB')
        image.show()
        raw_input("Second Image")


    #do all the things tested in the 'if' statement below
    def driver(self):
        ### temporary path for the images to be saved at. There's probably a better way to do this
        temp_path = './temp_im'
        generator = ic.rolling_window()
        count = 0
        ### generator function only returns one window at a time, potentially speeding image making
        for i in generator:
            self.log.info("rolling window num:{}".format(count))
            count += 1
            arr = self.convert_dataframe_to_np_array(df=i)
            arr = self.change_date(arr,count)
            self.log.info("creating/saving image...")
            self.create_image_from_np_array(arr,temp_path)
            self.log.info("reading image...")
            img_arr = self.read_image_to_np_array(temp_path)
            
            img_arr = self.delete_alpha(img_arr)
            labels = self.get_labels(arr,img_arr)
            img_arr = self.flatten_image(img_arr)
            img_arr = self.append_labels(img_arr,labels)
            self.log.info("saving array...")
            write_file = self.save_array(img_arr,count)
            self.recreate_image(write_file)
            ###Check image after deleting alpha array (need to adjust for flattened image)
            #img = Image.fromarray(arr,'RGB')
            #img.show()



if __name__ == '__main__':
    check_driver = 1
    days = [30,60,90]

    if check_driver == 1:
        ic = ImageCreator('store/nvda_100d.csv', 90,m=2)
        ic.driver()
    else:
        ic = ImageCreator('store/nvda_100d.csv',n = 90)
        # Default is 30 day window
        generator = ic.rolling_window()
        count = 0
        for i in generator:
            #self.log.info("rolling window num:{}".format(count))
            count += 1
            arr = ic.convert_dataframe_to_np_array(df=i)
            arr = ic.change_date(arr,count)
            #self.log.info("creating/saving image...")
            ic.create_image_from_np_array(arr,'temp_im')
            #self.log.info("reading image...")
            img_arr = ic.read_image_to_np_array('temp_im')
            img_arr = ic.delete_alpha(img_arr)
            image = Image.fromarray(img_arr,'RGB')
            image.show()
            raw_input("First Image")
            arr_shape = img_arr.shape
            #labels = ic.get_labels(arr,img_arr)
            img_arr = ic.flatten_image(img_arr)
            img_arr = list(img_arr)
            print "len1",len(img_arr)
            img_arr.append(arr_shape)
            print "len2",len(img_arr)
            #img_arr = np.delete(img_arr,[len(img_arr)-3,len(img_arr)-2,len(img_arr)-1])

            img_arr = np.asarray(img_arr[0:len(img_arr)-1])
            img_arr1 = np.reshape(img_arr,arr_shape)
            image = Image.fromarray(img_arr1,'RGB')
            image.show()
            raw_input("Second Image")
            #img_arr = ic.append_labels(img_arr,labels)
            #self.log.info("saving array...")
            #np.save('recreate_test',img_arr)
            #ic.recreate_image('recreate_test.npy',img_arr)
            raw_input("Press enter to continue")

            ###Check image after deleting alpha array (need to adjust for flattened image)
            #img = Image.fromarray(arr,'RGB')

            
        
