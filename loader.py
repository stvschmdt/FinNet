import pandas as pd

def load_ohlc_csv(filename):
	data = pd.read_csv(filename)
	return data
