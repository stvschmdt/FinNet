import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mpl_finance as fin
import logger
from PIL import Image
from mpl_finance import candlestick_ohlc
import os
### Test performance
import time


#format of API renamed from data_collector.py
#cols = ['low', 'open', 'high', 'close', 'volume', 'adj_close', 'split_coeff', 'dividend']
#class to read in csv from store plus operations
### Add symbol directory to imags_as_arrays, i.e. /imgs_as_arrays/amzn_5953d/30d, or 90d
class ImageCreator():
    def __init__(self, filename, n=30, m = 2,plot_dpi = 50):
        if filename != None:
            self.plot_dpi = plot_dpi #sets resolution of figure, figsize*dpi gives pixel dimensions
            self.percent_label_days = m
	    self.filename = filename
	    self.ndays = n
	    self.log = logger.Logging()
	    self.log.info('running for days: {}'.format(n))
            self.log.info('label days: {}'.format(m))
	    self.data = self.load_ohlc_csv()
            self.ohlc = self.reorder_data_columns()
            self.write_dir = self.check_dir()
            self.label_dir = self.write_dir
        else:
            self.plot_dpi = plot_dpi
            self.percent_label_days = m
            self.filename = filename
            self.ndays = n
            self.log = logger.Logging()
        
        print "Sample driver code"
        print (''' 
        generator = ic.rolling_window(); create a generator object containing ndays
        count = 0; set counter for filenames
        for i in generator:; loop through generator, each i is nday window
            count += 1
            arr = self.convert_dataframe_to_np_array(df=i)
            arr = self.change_date(arr,count); changes date string to 0-n integers
            img_arr = self.create_image_from_np_array(arr); creates image from window, saves to temp_path
            Note that this is handled in functions, but all pixel values must be dtype = uint8 to ensure images
            are not corrupted
            
            img_arr = self.delete_alpha(img_arr); removes alpha channel of image
            labels = self.get_labels(arr,img_arr); gets labels from data, labels are image shape and percent change m days out
            img_arr = self.flatten_image(img_arr); flattens array to 1D
            img_arr = self.append_labels(img_arr,labels); appends labels to 1D array
            write_file = self.save_array(img_arr,count); saves array to below:
            'write_file = self.write_dir + window(count)_label(self.percent_label_days)d'
            where self.write_dir is
            'self.write_dir = ./imgs_as_arrays/(symbol)/(n)/'
            and items wrapped in () are variables in code

            #self.recreate_image(write_file,imshow=True); method to recreate image from array, imshow=True displays image during runtime
            ''')

    def check_dir(self,filename=None,n=None,m=None):
        ### Check symbol directory and data directory
        ### if symbol not in directory, create new directory with symbol
        ### Check symbol directory for nday windows
        ### if no nday windows directory, create new with n + 'd'
        ### should check in symbol directories for files. If files have already been created, then skip to next symbol.
        ### this will allow continuous reading of the store directory for new ndays, symbols, and percent label days
        ## so that data is not being recreated and script can be run when new data is added.
        if filename == None:
            filename = self.filename
        if n == None:
            n = self.ndays
        if m == None:
            m = self.percent_label_days
        ### filename layout is 'store/symbol_numd.csv'
        ### 'num' + 'd' not to be confused with 'n' + 'd'
        try:
            _,symbol = filename.split("/")
            symbol,_ = symbol.split(".")
        except ValueError:
            self.log.error("given data filename is not in format 'store/symbol_numd.csv'")
        n = str(n)+'d'
        m = 'label'+str(m)+'d'
        ### Switch os.system calls to subprocess calls
        try:
            dir_len = len(os.listdir('./imgs_as_arrays/'+symbol+'/'+n+'/'+m+'/'))
            if dir_len < self.get_num_windows(self.ndays):
                write_dir = './imgs_as_arrays/'+symbol+'/'+n+'/'+m+'/'
            else:
                write_dir = './imgs_as_arrays/'+symbol+'/'+n+'/'+m+'/'
                self.log.info('skipping directory {}'.format(write_dir))
                write_dir = None

        except:
            os.system('mkdir -p ./imgs_as_arrays/'+symbol+'/'+n+'/'+m+'/')
            write_dir = './imgs_as_arrays/'+symbol+'/'+n+'/'+m+'/'
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
                    yield self.n_day(i,self.ndays+i,df=df)
                    
    def change_date(self,arr,count):
        ### Changes date string into index, adds count so that indices will be (for n = 30):
        ### 0-29 for window 0; 1-30 for window 1; 2-31 for window 2; etc
        for j in range(len(arr)):
            arr[j][0] = j - 1 + count
        return arr

    def create_image_from_np_array(self,arr, n=None, plot_dpi = None):
        if plot_dpi == None:
            plot_dpi = self.plot_dpi
        if n == None:
            n = self.ndays
        #create candlestick chart with matplotlib
        fig = plt.figure(figsize=(2,2),dpi=plot_dpi) #figsize can be edited to make bigger or smaller
        #ax1 = plt.gca()
        #x limits are the low and high indices, day 0 to day nday for window 1
        # y limits are the min and max of the ohlc data
        plt.xlim((arr[0,0],n+arr[0,0]))
        plt.ylim(np.min(arr[:,1:5]),np.max(arr[:,1:5]))
        #turn off axes around the plot
        plt.axis('off')
        plt.tight_layout()
        candlestick_ohlc(fig.axes[0],arr,colorup='#77d879', colordown='#db3f3f')
        ### dpi can be set as a parameter in class initiation, sets resolution
        #convert matplotlib figure to 4D numpy array (RGBA channels). Alpha channel (opacity) should be removed.
        #No need for writing or reading an image.

        #http://www.icare.univ-lille1.fr/wiki/index.php/How_to_convert_a_matplotlib_figure_to_a_numpy_array_or_a_PIL_image
        # draw the renderer
        fig.canvas.draw()
        #print fig.axes[0].get_children()
        # ax.get_children may allow optimization: see:
        #https://bastibe.de/2013-05-30-speeding-up-matplotlib.html

        # Get the RGBA Buffer from the figure
        w,h = fig.canvas.get_width_height()
        buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w,h,4)

        #canvas.tostring_argb give pixmap in ARGB mode. Roll the alpha channel to have it in RGBA mode
        buf = np.roll(buf,3,axis = 2)
        plt.close()
	return buf
    
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

    def flatten_image(self,arr):
        ### Flatten image to vector for easier saving and appending of label
	### 'C' parameter flattens C style, along rows, e.g. [[1,2,3],[4,5,6]] -> [1,2,3,4,5,6]
	### mapped as [i,j,k] -> [j + rowlen*i + k*rowlen*collen], but can be simply reshaped
        ### with np.reshape()
        return arr.flatten('C')

    def get_labels(self,arr,img,df = None):
        ### Calculate percentage change +m days after window
        ### Note that the entire data frame must be passed to df rather than a windowed data frame
        ### This is because the windowed frame doesn't contain the future price
        ### Will need shape in order to recreate image array
        if df is None:
            dataline = self.ohlc.iloc[arr[-1,0] + self.percent_label_days]
            percent_diff = (np.max(dataline[1:4]) - arr[-1,4])/np.max(dataline[1:4])
            return [img.shape[0],img.shape[1],img.shape[2]],np.float16(100*percent_diff)
        else:
            dataline = df.iloc[arr[-1,0] + self.percent_label_days]
            percent_diff = (np.max(dataline[1:4]) - arr[-1,4])/np.max(dataline[1:4])
            return [img.shape[0],img.shape[1],img.shape[2]],np.float16(100*percent_diff)

    def append_shape(self,img_arr,shape):
        ### Add label and shape to the end of flattened image array
        return np.append(img_arr,shape)

    def save_array(self,img_arr,count):
        ### create save path
        write_file = self.write_dir + 'window' + str(count)
        ### np.save creates a binary file that saves space over saving numbers; float16 saves space
        np.save(write_file,img_arr.astype('uint8'))
        ### return write file so image can be recreated
        return write_file

    def recreate_image(self,write_file,imshow=False):
        ### Recreate image from numpy file
        img_arr = np.load(write_file + '.npy')

        ### peel off labels and remove from image array
        shape = (int(img_arr[-3]),int(img_arr[-2]),int(img_arr[-1])) 
        img_arr = np.asarray(img_arr[:len(img_arr)-3].astype('uint8'))
        ### reshape image array from 1D to 3D
        img_arr = np.reshape(img_arr,shape)
        if imshow:
            ### display recreated image
            image = Image.fromarray(img_arr,'RGB')
            image.show()
            raw_input("Recreated image")
        return img_arr, percent_label

    def parse_recreate_image(self, filename, imshow = False):
        #in case we dont have write_file in memory - standalone function
        filename = filename.split('.')[0]
        row = self.recreate_image(filename, imshow)
        x = row[0]
        y = row[1]
        return x, y

    def parse_recreate_directory(self, directory, n_day='30d', d_out='2d'):
        #loop through directory to store all in np array tensor
	full_dir = directory + '/' + n_day +'/' + d_out
        files = os.listdir(full_dir)
        x_ = []
        y_ = []
        for f in files:
            row = parse_recreate_image(f)
            x_.append(row[0])
            y_.append(row[1])
        #return as np arrays ready to go
        x_ = np.array(x_)
        y_ = np.array(y_)
        return x_, y_

    def save_label(self,labels):
        print self.label_dir + 'yvals'
        np.save(self.label_dir+'yvals',labels)

    def __del__(self):
        self.log.info("Object deleted")

    #do all the things tested in the 'if' statement below
    def driver(self):
        ### temporary path for the images to be saved at. There's probably a better way to do this
        if self.write_dir == None:
            generator = []
        else:
            generator = ic.rolling_window()
        count = 0
        labels = np.asarray([])
        ### generator function only returns one window at a time, potentially speeding image making
        for i in generator:
            self.log.info("rolling window num:{}".format(count))
            count += 1
            arr = self.convert_dataframe_to_np_array(df=i)
            arr = self.change_date(arr,count)
            self.log.info("creating image...")
            img_arr = self.create_image_from_np_array(arr)
            img_arr = self.delete_alpha(img_arr)
            shape,label = self.get_labels(arr,img_arr)
            labels = np.append(labels,label)
            arr = None
            img_arr = self.flatten_image(img_arr)
            img_arr = self.append_shape(img_arr,shape)
            self.log.info("saving array...")
            write_file = self.save_array(img_arr,count)
            img_arr = None
            #self.recreate_image(write_file,imshow=True)
        if self.label_dir == None:
            pass
        else:
            self.save_label(labels)



if __name__ == '__main__':
    check_driver = 1
    days = [30,60,90]
    percent_label_days = [1,2,5]
    label_arr = np.asarray([])
    if check_driver == 1:
        '''
        for filename in os.listdir('store/'):
            str_filename = filename
            t0 = time.time()
            for day in days:
                for label_day in percent_label_days:
                    ic = ImageCreator('store/'+str_filename, n=day,m=label_day,plot_dpi = 25)
                    ic.driver()
                    del ic
        '''
        ic = ImageCreator('store/ba_5953d.csv', n=90,m=1,plot_dpi = 25)
        ic.driver()

    else:
        pass
        '''
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
            '''
            
        
