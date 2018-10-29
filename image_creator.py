import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import mpl_finance as fin
import logger
from PIL import Image
from mpl_finance import candlestick_ohlc
import os
import csv
### Test performance
import time


#format of API renamed from data_collector.py
#cols = ['low', 'open', 'high', 'close', 'volume', 'adj_close', 'split_coeff', 'dividend']
#class to read in csv from store plus operations
'''
ImageCreator: 
    reads in csv, ndays (window size), m days out (days out from last window to search for max price, and plot_dpi (sets resolution)
    csv is set to pandas dataframe
    check_dir searches through directories, to create write directory if needed and set write_dir variable.
    If length of data frame is equal to or greater than the length of the chosen directory, IC will skip the directory
    pandas data frame is turned into numpy array
    numpy array is used to generate all needed candle stick objects, patches (rectangles) and lines
    numpy array generates first window, from data line 0:nday, i.e. 0 to 29 inclusive for nday = 30
    the candle stick objects from the first window are gathered in the variables 'patches' and 'lines'
    the first plot is turned into pixel data and saved
    the loop begins by finding the next candle stick objects and getting the x,y, width* and height*, and color
    * = patches only
    it then replaces the object going out of the window with the new plot data, i.e. for n = 30 the first loop
    will shift the object data, i.e. for lines, (Line1 2D, Line2 2D,..., Line29 2D) to (Line30 2D, Line 1 2D, ..., Line 29 2D).
    The order of the objects is irrelevant, given that the position on the graph is set by x and y. It continues updating
    plot data in this fashion, gathering the image pixel data, and saving.
    
    Throughout the loop, labels are gathered from the data 'end of window'+1:mdays+'end of window', inclusive. They are appended to 
    a separate array, and then saved to the write_dir as yvals.npy.

    All files saved are numpy binary files, .npy, and must be read back in by numpy. 

    The parse recreate directory function recreates all of the images and corresponding yvals for a given directory.
    Note that to recreate the image, the dtype of the numpy array must be uint8, which is automated in the function recreate_image.

    '''
class ImageCreator():
    def __init__(self):
        self.control_file = 'config_imgs.txt'
        config = self.read_control_file(self.control_file)
        self.use_max = bool(int(config['use_max'][0]))
        self.fig_dim = int(config['fig_dim'][0]) #sets the figsize, figsize*dpi gives pixel dimensions
        self.plot_dpi = int(config['plot_dpi'][0]) #sets resolution of figure, figsize*dpi gives pixel dimensions
        self.percent_label_days = map(int,config['outlook'][0].split(','))
        self.labels_only = bool(int(config['labels_only'][0]))
        self.filenames = config['symbols'][0].split(',')
	self.ndays = map(int,config['ndays'][0].split(','))
        self.normalized = bool(int(config['normalized'][0]))
	self.log = logger.Logging()
        self.data = self.load_ohlc_csv('store/amd_100d.csv')
        self.ohlc = self.reorder_data_columns()
        
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

    def check_dir(self,filename=None,n=None,m=None,df = None):
        ### Check symbol directory and data directory
        ### if symbol not in directory, create new directory with symbol
        ### Check symbol directory for nday windows
        ### if no nday windows directory, create new with n + 'd'
        ### should check in symbol directories for files. If files have already been created, then skip to next symbol.
        ### this will allow continuous reading of the store directory for new ndays, symbols, and percent label days
        ## so that data is not being recreated and script can be run when new data is added.
        if df is None:
            df = self.ohlc
        if filename == None:
            filename = self.filenames[0]
        if n == None:
            n = self.ndays[0]
        if m == None:
            m = self.percent_label_days[0]
        ### filename layout is 'store/symbol_numd.csv'
        ### 'num' + 'd' not to be confused with 'n' + 'd'
        try:
            _,symbol = filename.split("/")
            symbol,_ = symbol.split(".")
        except ValueError:    
            self.log.error("given data filename is not in format 'store/symbol_numd.csv'")
        n_s = str(n)+'d'
        m_s = str(m)+'d'
        try:
            dir_len = len(os.listdir('./imgs_as_arrays/'+symbol+'/'+n_s+'/'+m_s+'/'))
            if dir_len < self.get_num_windows(n = n,m = m,df = df) or self.labels_only:
                write_dir = './imgs_as_arrays/'+symbol+'/'+n_s+'/'+m_s+'/'
            else:
                write_dir = './imgs_as_arrays/'+symbol+'/'+n_s+'/'+m_s+'/'
                self.log.info('skipping directory {}'.format(write_dir))
                write_dir = None

        except:
            os.system('mkdir -p ./imgs_as_arrays/'+symbol+'/'+n_s+'/'+m_s+'/')
            write_dir = './imgs_as_arrays/'+symbol+'/'+n_s+'/'+m_s+'/'
        return write_dir
                

    def load_ohlc_csv(self,filename):
	try:
	    data = pd.read_csv(filename)
	    return data
	except:
	    self.log.error('could not read file {}'.format(filename))

    def reorder_data_columns(self,data = None):
        try:
            if data is None:
                data = self.data
            ohlc = data[['date','open','high','low','adj_close']]
            return ohlc
        except:
            self.log.error('could not reorder data columns')

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
        #try:
        control = {}
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter=':')
            for row in reader:
                control[row[0]] = row[1:]
        return control
        #except:
           # self.log.error('failed to process control file')


    def n_day(self,window_start,window_stop, df=None):
        #return either n day window from self.data or pass in dataframe
	if df is None:
	    return self.ohlc.iloc[window_start:window_stop]
	else:
	    return df.iloc[window_start:window_stop]

    def normalize_data(self,df = None):
        ### take full dataframe data and normalize by adjusted close,
        ### i.e. open -> open/(close/adj_close)
        ### Put flags; if adj_
        ### imgs config file, for f in list of stocks, days,adjusted flag
        if df is None:
            df = self.data
        norm = df['close']/df['adj_close']
        df['open'] = df['open']/norm
        df['high'] = df['high']/norm
        df['low'] = df['low']/norm
        return df



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

    def get_num_windows(self,n=None,m = None,df=None):
        ### When appending label, cannot put label for windows where index is out of range
        ###if n = 90, cannot append 2 days out label for window 9-98 because index 100 is
        ### out of range for data frame, so we subtract self.percent_label_days and add 1 to number of windows
        if m == None:
            m = self.percent_label_days[0]
        if n == None:
            n = self.ndays[0]
        if df is None:
	    return len(self.ohlc) - n - m + 1
        else:
            return len(df) - n - m + 1


    def create_image_from_numpy_array(self,arr_full,num_windows,plot_dpi = None, n=None,fig_dim = None):
        if n == None:
            n = self.ndays
        if fig_dim == None:
            fig_dim = self.fig_dim
        if plot_dpi == None:
            plot_dpi = self.plot_dpi
        fig_alldata = plt.figure(figsize=(fig_dim,fig_dim),dpi=plot_dpi) 
        ax_alldata = plt.gca()
        plt.xlim((0,num_windows))
        plt.ylim(np.min(arr_full[:,1:5]),np.max(arr_full[:,1:5]))
        candle_stick = candlestick_ohlc(ax_alldata,arr_full[n:],colorup='#77d879', colordown='#db3f3f')
        lines = candle_stick[0]
        patches = candle_stick[1]

        plt.clf()
        plt.cla()
        plt.close()
        return lines, patches


    def create_figure_instance(self,arr_full,n=None,plot_dpi = None):
        if n == None:
            n = self.ndays[0]
        if plot_dpi == None:
            plot_dpi = self.plot_dpi
        fig_windows = plt.figure(figsize=(2,2),dpi = plot_dpi)
        ax_windows = plt.gca()
        plt.xlim((-0.2,n-0.8))
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

    def get_labels(self,arr,img,df = None,use_max = None,n = None,m = None):
        ### Calculate percentage change +m days after window
        ### Note that the entire data frame must be passed to df rather than a windowed data frame
        ### This is because the windowed frame doesn't contain the future price
        ### Will need shape in order to recreate image array
        if n == None:
            n = self.ndays[0]
        if m == None:
            m = self.percent_label_days[0]
        if use_max == None:
            use_max = self.use_max

        if use_max:
            def return_maxmin(datalines):
                return np.max(datalines)
        else:
            def return_maxmin(datalines):
                return np.min(datalines)


        if df is None:
            datalines = self.ohlc.iloc[arr[-1,0]+1 : arr[-1,0] + m+1,:]
            datalines = self.convert_dataframe_to_np_array(df=datalines)
            percent_diff = (return_maxmin(datalines[:,1:5]) - arr[-1,4])/arr[-1,4]
            return [img.shape[0],img.shape[1],img.shape[2]],np.float16(percent_diff)
        else:
            datalines = df.iloc[arr[-1,0]+1:arr[-1,0] + m+1]
            datalines = self.convert_dataframe_to_np_array(df=datalines)
            percent_diff = (return_maxmin(datalines[:,1:5]) - arr[-1,4])/arr[-1,4]
            return [img.shape[0],img.shape[1],img.shape[2]],np.float16(percent_diff)

    def append_shape(self,img_arr,shape):
        ### Add label and shape to the end of flattened image array
        return np.append(img_arr,shape)

    def save_array(self,write_dir,img_arr,count):
        ### create save path
        write_file = write_dir + str(count)
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

    def sort_files(self,files,use_max = None):
        if use_max == None:
            use_max = self.use_max
        if use_max:
            minmax = 'maxy'
        else:
            minmax = 'miny'
        nums = []
        img_arrs = []
        for f in files:
            if minmax not in f:
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

    def parse_recreate_directory(self, directory, n_day = '30d', d_out = '2d', use_max = None):
        #loop through directory to store all in np array tensor
        if use_max == None:
            use_max = self.use_max
        if use_max:
            minmax = 'maxy'
        else:
            minmax = 'miny'
        full_dir = directory + '/' + n_day + '/' + d_out + '/'
        files = os.listdir(full_dir)
        ###os.listdir does not read arrays in window order, so we must sort them
        files = self.sort_files(files)
        x_ = []
        y_ = []
        for f in files:
            if minmax not in f:
                row = self.recreate_image(full_dir + f)
                x_.append(row)
            else:
                y_ = np.load(full_dir + f)
        #return as np arrays ready to go
        x_ = np.array(x_)
        y_ = np.array(y_)
        return x_, y_

    def save_label(self,label_dir,labels,use_max = None):
        if use_max == None:
            use_max = self.use_max
        if use_max:
            minmax = 'maxy'
        else:
            minmax = 'miny'
        np.save(label_dir+minmax,labels)

    def update_current_data(self,candle_stick,newlinedata,newlinecolor,newpatchdata,count,n = None):
        if n == None:
            n = self.ndays[0]
        if count >= n:
            count = count % n
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

    def redraw_image(self,candle_stick,fig,ax,count,arr,n = None):
        if n == None:
            n = self.ndays[0]
        for line in candle_stick[0]:
            ax.draw_artist(line)
        for patch in candle_stick[1]:
            ax.draw_artist(patch)
        #fig.canvas.draw_idle()
        plt.xlim((count-0.2,count+n-0.8))
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


    def get_current_window(self,arr,count,n = None):
        if n == None:
            n = self.ndays[0]
        return arr[count:count+n]


    def driver(self):
        for symbol in self.filenames:
            filepath = 'store/'+symbol+'.csv'
            for n in self.ndays:
                for m in self.percent_label_days:
                    self.log.info('running for symbol: {}'.format(filepath))
	            self.log.info('running for days: {}'.format(n))
                    self.log.info('label days: {}'.format(m))

	            data = self.load_ohlc_csv(filepath)
                    if self.normalized:
                        data = self.normalize_data(df = data)
                    ohlc = self.reorder_data_columns(data = data)

                    write_dir = self.check_dir(filename=filepath,n = n, m = m, df = ohlc)
                    label_dir = write_dir
                    if write_dir == None and not self.labels_only:
                        self.log.info('skipping directory...')
                        continue
                        

                    arr_full = self.convert_dataframe_to_np_array(df = ohlc)
                    arr_full = self.change_date(arr_full,0)
                    num_windows = self.get_num_windows(m = m, n = n, df = ohlc)
                    self.log.info('creating figure from full dataframe...')
                    if not self.labels_only:
                        lines, patches = self.create_image_from_numpy_array(arr_full,num_windows,n = n)
                        ### figure for windows of nday data
                        self.log.info('creating first windowed figure...')
                    candle_stick,fig_windows,ax_windows,img_arr = self.create_figure_instance(arr_full,n = n)
                    img_arr = self.delete_alpha(img_arr)
                    shape, label = self.get_labels(arr_full[0:n],img_arr,df = ohlc)
                    labels = np.asarray([label])
                    if not self.labels_only:
                        img_arr = self.flatten_image(img_arr)
                        img_arr = self.append_shape(img_arr,shape)
                        self.log.info('saving array...')
                        write_file = self.save_array(write_dir,img_arr,0)


                    for i in range(1,num_windows):
                        self.log.info("window {} ".format(i))
                        arr = self.get_current_window(arr_full,i,n = n)
                        self.log.info('getting next plot data...')
                        if not self.labels_only:
                            newlinedata, newlinecolor, newpatchdata = self.get_new_plot_data(lines,patches,i)
                            self.log.info('updating line data...')
                            candle_stick = self.update_current_data(candle_stick,newlinedata,newlinecolor,newpatchdata,i,n = n)
                            self.log.info('redrawing image...')
                            img_arr = self.redraw_image(candle_stick,fig_windows,ax_windows,i,arr,n = n)
                            img_arr = self.delete_alpha(img_arr)

                        shape, label = self.get_labels(arr,img_arr,df = ohlc)
                        labels = np.append(labels,label)
                        if not self.labels_only:
                            img_arr = self.flatten_image(img_arr)
                            img_arr = self.append_shape(img_arr,shape)
                            self.log.info("saving array...")
                            #write_file = self.save_array(write_dir,img_arr,i)
                        #self.recreate_image(write_file,imshow = True)
                    if label_dir == None and not self.labels_only:
                        pass
                    else:
                        self.log.info('saving yvals...')
                        self.save_label(label_dir,labels)
                    ### Close plot at end
                    plt.close()


    

    def __del__(self):
        self.log.info("Object deleted")

if __name__ == '__main__':
    check_driver = 1
    if check_driver == 1:
        t0 = time.time()
        ic = ImageCreator()
        ic.driver()
        print time.time() - t0
    else:
        ic = ImageCreator('store/amd_100d.csv', n=30,m=2,plot_dpi = 25,use_max = False)
        control = ic.read_control_file()
        print map(int,control['days'][0].split(','))
