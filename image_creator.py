import pandas as pd
import matplotlib
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
    def __init__(self, filename=None, n=30, m = 2,plot_dpi = 50):
        if filename != None:
            self.plot_dpi = plot_dpi #sets resolution of figure, figsize*dpi gives pixel dimensions
            self.percent_label_days = m
	    self.filename = filename
	    self.ndays = n
	    self.log = logger.Logging()
            self.log.info('running for symbol: {}'.format(filename))
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

    def driver(self):
        arr_full = self.convert_dataframe_to_np_array()
        arr_full = self.change_date(arr_full,0)
        #lines,patches = ic.create_image_from_np_array(arr)
        num_windows = self.get_num_windows()
        self.log.info('creating figure from full dataframe...')
        lines, patches = self.create_image_from_numpy_array(arr_full,num_windows)
        ### figure for windows of nday data
        self.log.info('creating first windowed figure...')
        candle_stick,fig_windows,ax_windows,img_arr = self.create_figure_instance(arr_full)
        img_arr = self.delete_alpha(img_arr)
        ### Check this
        shape, label = self.get_labels(arr_full[0:self.ndays],img_arr)
        labels = np.asarray([label])
        img_arr = self.flatten_image(img_arr)
        img_arr = self.append_shape(img_arr,shape)
        if self.write_dir == None:
            ### skip directory
            self.log.info('skipping directory...')
            num_windows = 1
        else:
            self.log.info('saving array...')
            write_file = self.save_array(img_arr,0)


        for i in range(1,num_windows):
            self.log.info("window {} ".format(i))
            arr = ic.get_current_window(arr_full,i)
            self.log.info('getting next plot data...')
            newlinedata, newlinecolor, newpatchdata = ic.get_new_plot_data(lines,patches,i)
            self.log.info('updating line data...')
            candle_stick = ic.update_current_data(candle_stick,newlinedata,newlinecolor,newpatchdata,i)
            self.log.info('redrawing image...')
            img_arr = ic.redraw_image(candle_stick,fig_windows,ax_windows,i,arr)

            img_arr = self.delete_alpha(img_arr)
            shape, label = self.get_labels(arr,img_arr)
            labels = np.append(labels,label)
            img_arr = self.flatten_image(img_arr)
            img_arr = self.append_shape(img_arr,shape)

            self.log.info("saving array...")
            write_file = self.save_array(img_arr,i)
            'write_file = self.write_dir + window(count)_label(self.percent_label_days)d'
            where self.write_dir is
            'self.write_dir = ./imgs_as_arrays/(symbol)/(n)/'
            and items wrapped in () are variables in code

            #self.recreate_image(write_file,imshow = True)

        if self.label_dir == None:
            label_dir set by write_dir: if write dir is full then label_dir will pass over saving
            pass
        else:
            self.log.info('saving yvals...')
            self.save_label(labels)
        ### Close plot at end
        plt.close()


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
        m = str(m)+'d'
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
        

    def change_date(self,arr,count):
        ### Changes date string into index, adds count so that indices will be (for n = 30):
        ### 0-29 for window 0; 1-30 for window 1; 2-31 for window 2; etc
        for j in range(len(arr)):
            arr[j][0] = j  + count
        return arr

    def get_num_windows(self,n=None,df=None):
        ### When appending label, cannot put label for windows where index is out of range
        ###if n = 90, cannot append 2 days out label for window 9-98 because index 100 is
        ### out of range for data frame, so we subtract self.percent_label_days and add 1 to number of windows
        if n == None:
            n = self.ndays
        if df is None:
	    return len(self.ohlc) - n - self.percent_label_days + 1
        else:
            return len(df) - n - self.percent_label_days + 1


    def create_image_from_numpy_array(self,arr_full,num_windows,plot_dpi = None, n=None):
        if n == None:
            n = self.ndays
        if plot_dpi == None:
            plot_dpi = self.plot_dpi
        fig_alldata = plt.figure(figsize=(2,2),dpi=25) 
        ax_alldata = plt.gca()
        plt.xlim((0,num_windows))
        plt.ylim(np.min(arr_full[:,1:5]),np.max(arr_full[:,1:5]))
        candle_stick = candlestick_ohlc(ax_alldata,arr_full[self.ndays:],colorup='#77d879', colordown='#db3f3f')
        lines = candle_stick[0]
        patches = candle_stick[1]

        plt.clf()
        plt.cla()
        plt.close()
        return lines, patches


    def create_figure_instance(self,arr_full,n=None,plot_dpi = None):
        if n == None:
            n = self.ndays
        if plot_dpi == None:
            plot_dpi = self.plot_dpi
        fig_windows = plt.figure(figsize=(2,2),dpi = plot_dpi)
        ax_windows = plt.gca()
        plt.xlim((-0.2,ic.ndays-0.8))
        plt.ylim(np.min(arr_full[:n,1:5]),np.max(arr_full[:n,1:5]))
        plt.axis('off')
        plt.tight_layout()
        plt.ion()
        # next data needs axes to plot to. get data from this
        ### set first window
        candle_stick = candlestick_ohlc(ax_windows,arr_full[:n],colorup='#77d879', colordown='#db3f3f')
        fig_windows.canvas.draw()
        ### create first image array below
        # Get the RGBA Buffer from the figure
        w,h = fig_windows.canvas.get_width_height()
        buf = np.fromstring(fig_windows.canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w,h,4)

        #canvas.tostring_argb give pixmap in ARGB mode. Roll the alpha channel to have it in RGBA mode
        buf = np.roll(buf,3,axis = 2)
        return candle_stick,fig_windows,ax_windows,buf

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

    def get_labels(self,arr,img,df = None,use_max = True):
        ### Calculate percentage change +m days after window
        ### Note that the entire data frame must be passed to df rather than a windowed data frame
        ### This is because the windowed frame doesn't contain the future price
        ### Will need shape in order to recreate image array
        if use_max:
            def return_maxmin(datalines):
                return np.max(datalines)
        else:
            def return_maxmin(datalines):
                return np.min(datalines)

        if df is None:
            datalines = self.ohlc.iloc[arr[-1,0]+1 : arr[-1,0] + self.percent_label_days+1,:]
            datalines = self.convert_dataframe_to_np_array(df=datalines)
            percent_diff = (return_maxmin(datalines[:,1:5]) - arr[-1,4])/arr[-1,4]
            return [img.shape[0],img.shape[1],img.shape[2]],np.float16(percent_diff)
        else:
            datalines = df.iloc[arr[-1,0]+1:arr[-1,0] + self.percent_label_days+1]
            datalines = self.convert_dataframe_to_np_array(df=datalines)
            percent_diff = (return_maxmin(datalines[:,1:5]) - arr[-1,4])/arr[-1,4]
            return [img.shape[0],img.shape[1],img.shape[2]],np.float16(percent_diff)

    def append_shape(self,img_arr,shape):
        ### Add label and shape to the end of flattened image array
        return np.append(img_arr,shape)

    def save_array(self,img_arr,count):
        ### create save path
        write_file = self.write_dir + str(count)
        ### np.save creates a binary file that saves space over saving numbers; float16 saves space
        np.save(write_file,img_arr.astype('uint8'))
        ### return write file so image can be recreated
        return write_file

    def recreate_image(self,write_file,imshow=False):
        ### Recreate image from numpy file
        try:
            img_arr = np.load(write_file)
        except:
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
        return img_arr

    def sort_files(self,files):
        nums = []
        img_arrs = []
        for f in files:
            if 'yvals' not in f:
                img_arrs.append(f)
                file_stem = f.split('.')[0]
                nums.append(int(file_stem))
            else:
                yval_file = f
        z = [x for _,x in sorted(zip(nums,img_arrs))]
        z.append(yval_file)
        return z

    def parse_recreate_image(self, filename, imshow = False):
        #in case we dont have write_file in memory - standalone function
        filename = filename.split('.')[0]
        row = self.recreate_image(filename, imshow)
        x = row[0]
        y = row[1]
        return x, y

    def parse_recreate_directory(self, directory, n_day = '30d', d_out = '2d'):
        #loop through directory to store all in np array tensor
        full_dir = directory + '/' + n_day + '/' + d_out + '/'
        files = os.listdir(full_dir)
        ###os.listdir does not read arrays in window order, so we must sort them
        files = self.sort_files(files)
        x_ = []
        y_ = []
        for f in files:
            if 'yvals' not in f:
                row = self.recreate_image(full_dir + f)
                x_.append(row)
            else:
                y_ = np.load(full_dir + f)
        #return as np arrays ready to go
        x_ = np.array(x_)
        y_ = np.array(y_)
        return x_, y_

    def save_label(self,labels):
        np.save(self.label_dir+'yvals',labels)

    def update_current_data(self,candle_stick,newlinedata,newlinecolor,newpatchdata,count):
        if count >= self.ndays:
            count = count % self.ndays
        ### Set new line data
        newlinedata = np.transpose(newlinedata)
        candle_stick[0][count-1].set_data(newlinedata)
        candle_stick[0][count-1].set_color(newlinecolor)
        ### set new patch data
        candle_stick[1][count-1].set_x(newpatchdata[0])
        candle_stick[1][count-1].set_y(newpatchdata[1])
        candle_stick[1][count-1].set_width(newpatchdata[2])
        candle_stick[1][count-1].set_height(newpatchdata[3])
        candle_stick[1][count-1].set_color(newpatchdata[4])
        #candle_stick[0] = np.roll(candle_stick[0],self.ndays - 1)
        #candle_stick[1] = np.roll(candle_stick[1],self.ndays - 1)
        return candle_stick

    def get_new_plot_data(self,lines,patches,count):
        newline = lines[count].get_xydata()
        newlinecolor = lines[count].get_color()
        newpatch_xloc,newpatch_yloc = patches[count].get_xy() 
        newpatch_width = patches[count].get_width()
        newpatch_height = patches[count].get_height()
        newpatch_color = patches[count].get_facecolor()
        newpatch = [newpatch_xloc, newpatch_yloc, newpatch_width, newpatch_height,newpatch_color]
        return newline, newlinecolor, newpatch


    def reorder_current_data(self,candle_stick):
        current_lines = candle_stick[0]
        current_patches = candle_stick[1]
        return np.roll(current_lines,len(current_lines) - 1), np.roll(current_patches, len(current_patches) -1)

    def redraw_image(self,candle_stick,fig,ax,count,arr):
        for line in candle_stick[0]:
            ax.draw_artist(line)
        for patch in candle_stick[1]:
            ax.draw_artist(patch)
        #fig.canvas.draw_idle()
        plt.xlim((count-0.2,count+self.ndays-0.8))
        plt.ylim(np.min(arr[:,1:5]),np.max(arr[:,1:5]))
        #plt.axis('off')
        #plt.tight_layout()
        #fig.canvas.update()
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        #fig.canvas.draw()
        # Get the RGBA Buffer from the figure
        w,h = fig.canvas.get_width_height()
        buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w,h,4)

        #canvas.tostring_argb give pixmap in ARGB mode. Roll the alpha channel to have it in RGBA mode
        buf = np.roll(buf,3,axis = 2)
        return buf


    def get_current_window(self,arr,count):
        return arr[count:count+self.ndays]


    def driver(self):
        arr_full = self.convert_dataframe_to_np_array()
        arr_full = self.change_date(arr_full,0)
        #lines,patches = ic.create_image_from_np_array(arr)
        num_windows = self.get_num_windows()
        self.log.info('creating figure from full dataframe...')
        lines, patches = self.create_image_from_numpy_array(arr_full,num_windows)
        ### figure for windows of nday data
        self.log.info('creating first windowed figure...')
        candle_stick,fig_windows,ax_windows,img_arr = self.create_figure_instance(arr_full)
        img_arr = self.delete_alpha(img_arr)
        ### Check this
        shape, label = self.get_labels(arr_full[0:self.ndays],img_arr)
        labels = np.asarray([label])
        img_arr = self.flatten_image(img_arr)
        img_arr = self.append_shape(img_arr,shape)
        if self.write_dir == None:
            ### skip directory
            self.log.info('skipping directory...')
            num_windows = 1
        else:
            self.log.info('saving array...')
            write_file = self.save_array(img_arr,0)


        for i in range(1,num_windows):
            self.log.info("window {} ".format(i))
            arr = ic.get_current_window(arr_full,i)
            self.log.info('getting next plot data...')
            newlinedata, newlinecolor, newpatchdata = ic.get_new_plot_data(lines,patches,i)
            self.log.info('updating line data...')
            candle_stick = ic.update_current_data(candle_stick,newlinedata,newlinecolor,newpatchdata,i)
            self.log.info('redrawing image...')
            img_arr = ic.redraw_image(candle_stick,fig_windows,ax_windows,i,arr)

            img_arr = self.delete_alpha(img_arr)
            shape, label = self.get_labels(arr,img_arr)
            labels = np.append(labels,label)
            img_arr = self.flatten_image(img_arr)
            img_arr = self.append_shape(img_arr,shape)
            self.log.info("saving array...")
            write_file = self.save_array(img_arr,i)
            #self.recreate_image(write_file,imshow = True)
        if self.label_dir == None:
            pass
        else:
            self.log.info('saving yvals...')
            self.save_label(labels)
        ### Close plot at end
        plt.close()


    

    def __del__(self):
        self.log.info("Object deleted")

if __name__ == '__main__':
    check_driver = 1
    days = [30,60,90]
    percent_label_days = [1,2,5]
    label_arr = np.asarray([])
    if check_driver == 1:
        for d in days:
            ic = ImageCreator('store/amd_100d.csv', n=d,m=2,plot_dpi = 25)
            ic.driver()
        
        '''     
        for filename in os.listdir('store/'):
            str_filename = filename
            for day in days:
                for label_day in percent_label_days:
                    ic = ImageCreator('store/'+str_filename, n=day,m=label_day,plot_dpi = 25)
                    ic.driver()
        '''            
    else:
        pass
