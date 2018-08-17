import matplotlib.pyplot as plt
from matplotlib.finance import candlestick2_ohlc
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

def read_csv(filename, t_conv=True):
    df = pd.read_csv(filename).dropna()

    if t_conv:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'],unit='s')
    return df

def create_save_cs(df, test=False):
    fig, ax = plt.subplots()
    candlestick2_ohlc(ax, df['Open'], df['High'], df['Low'], df['Close'], width=0.6)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    if test:
        fig.autofmt_xdate()
        fig.tight_layout()
        plt.show()
    else:
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.canvas.draw()
    # Now we can save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return data

def process_dataframe(df, window_size):
    l_x_vals = []
    for i in xrange(len(df)-window_size):
        print 'processing:' , i, i+window_size-1
        l_x_vals.append(create_save_cs(df[i:window_size]))
    return l_x_vals

if __name__ == '__main__':
    data = read_csv('/home/ubuntu/store/crypto/coinbaseUSD_1-min_data_2014-12-01_to_2017-05-31.csv')
    #print data[:1]
    #print data[30:59]
    #cs = create_save_cs(data[-30:])
    #print cs
    x_vals = process_dataframe(data[500000:], 1440)
    save_data = pd.DataFrame(x_vals)
    save_data.to_csv('btccoindata.csv')
    print x_vals[0]
# Make a random plot...
#fig = plt.figure()
#fig.add_subplot(111)

# If we haven't already shown or saved the plot, then we need to
# draw the figure first...
#fig.canvas.draw()

# Now we can save it to a numpy array.
#data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
#data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
