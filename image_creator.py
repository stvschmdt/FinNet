import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.finance as fin
import logger
from PIL import Image
from matplotlib.finance import candlestick_ohlc

#class to read in csv from store plus operations
class ImageCreator():
    def __init__(self, filename, n=30, m = 2):
        self.percent_label_days = m
	self.filename = filename
	self.ndays = n
	self.log = logger.Logging()
	self.log.info('running for days: {}'.format(n))
	self.data = self.load_ohlc_csv()
        #self.data = self.reorder_data_columns(self.data)
        ### Initialize dictionary to order columns
        self.coldict = {
                'date':0,
                'Date':0,
                'open':1,
                'Open':1,
                'high':2,
                'High':2,
                'low':3,
                'Low':3,
                'adj_close':4,
                'Adj_Close':4,
                'Adjusted_close':4,
                'Adjusted_Close':4
                }

    def load_ohlc_csv(self):
	try:
	    data = pd.read_csv(self.filename)
	    return data
	except:
	    self.log.error('could not read file {}'.format(self.filename))

    def reorder_data_columns(self,data):
        try:
            collist = data.columns.values
            ### Create empty array to put integer values to sort by in
            sorter_array = []
            for col in collist:
                if col in self.coldict:
                    ### Get integer values from dictionary lookup
                    sorter_array.append(self.coldict[col])
                else:
                    ### Drop columns that do not match dictionary
                    ### May drop needed columns; could 
                    ### add a way to add columns during runtimethat may be needed
                    data = data.drop(columns = col)
            collist = data.columns.values
            new_collist = [x for _,x in sorted(zip(sorter_array,collist))]
            data = data[new_collist]
            return data
        except:
            self.log.error('could not reorder data columns')

    def n_day(self,window_start,window_stop, df=None):
        #return either n day window from self.data or pass in dataframe
	if df is None:
	    return self.data.iloc[window_start:window_stop]
	else:
	    return df.iloc[window_start:window_stop]

    def convert_dataframe_to_np_array(self,df=None):
        ### returns values of dataframe for numpy array
        if df is None:
            return self.data.values
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

    def create_image_from_np_array(self,arr,savepath):
        #create candlestick chart with matplotlib
        fig = plt.figure()
        ax1 = plt.gca()
        #x limits are the low and high indices, day 0 to day nday for window 1
        # y limits are the min and max of the ohlc data
        plt.xlim(arr[0][0],arr[-1][0])
        plt.ylim(np.min(arr[:,1:5]),np.max(arr[:,1:5]))
        #turn off axes around the plot
        plt.axis('off')
        plt.tight_layout()
        candlestick_ohlc(ax1,arr,colorup='#77d879', colordown='#db3f3f')
        ### Saving image here because I'm not sure how to pass to save_image function
        ### Maybe pass candlestick object?
        plt.savefig(savepath+'.png')
#       ax1.set_facecolor('w')
        #plt.Axes(fig,[0,0,1,1])
    
    def save_image(self,im, im_filename):
	return 0

    def get_num_windows(self,n,df=None):
        ### When appending label, cannot put label for windows where index is out of range
        ###if n = 90, cannot append 2 days out label for window 9-98 because index 100 is
        ### out of range for data frame, so we subtract self.percent_label_days and add 1 to number of windows
        if df is None:
	    return len(self.data) - n - self.percent_label_days + 1
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
	arr1 = arr.flatten('C')
        return arr1
    def get_labels(self,arr,img,df = None):
        ### Calculate percentage change +m days after window
        ### Note that the entire data frame must be passed to df rather than a windowed data frame
        ### This is because the windowed frame doesn't contain the future price
        ### Will need shape in order to recreate image array
        if df is None:
            dataline = self.data.iloc[arr[-1][0] + self.percent_label_days]
            percent_diff = (np.max(dataline[1:4]) - self.data.iloc[-1][4])/np.max(dataline[1:4])
            return [img.shape[0],img.shape[1],img.shape[2],int(100*percent_diff)]
        else:
            dataline = df.iloc[arr[-1][0] + self.percent_label_days]
            percent_diff = (np.max(dataline[1:4]) - df.iloc[-1][4])/np.max(dataline[1:4])
            return [img.shape[0],img.shape[1],img.shape[2],int(100*percent_diff)]
    def append_labels(self,arr,labels):
        ### Add label and shape to the end of flattened image array
        arr = np.append(arr,labels)
        return arr

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
	    np.save('./imgs_as_arrays/img_'+str(self.ndays)+'_days_window_'+str(count),img_arr)
            ###Check image after deleting alpha array (need to adjust for flattened image)
            #img = Image.fromarray(arr,'RGB')
            #img.show()



if __name__ == '__main__':
    check_driver = 0
    days = [30,60,90]

    if check_driver == 1:
        ic = ImageCreator('store/goog_100d.csv', 90,m=2)
        ic.driver()
    else:
        '''
        ic = ImageCreator('./store/goog_100d.csv',n = 90)
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
                arr[i][0] = i - 1 + count
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
            plt.close()
            #plt.show()
            img = np.asarray(Image.open('./test1.png'))
            ### Test mapping from 1D array back to 3D array to recreate image array
            label = ic.get_labels(arr,img)
            
            img1 = ic.flatten_image(img)
            img1 = ic.append_labels(img1,label)
           '''
    
        ### Test column reordering
        ic = ImageCreator('./store/lrcx_5953d.csv',n = 90)
        data = ic.reorder_data_columns(ic.data)
        print data.columns.values
    

        
