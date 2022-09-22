# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Built-in libraries
#
import numpy    as np
import pandas   as pd
from   datetime import date
from   datetime import timedelta


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Scipy library
#
from scipy.stats import skew
from scipy.stats import kurtosis





# 
# Utility function for feature engineering
#


def get_timespan(df, today, days):    
    df = df[pd.date_range(today - timedelta(days=days), 
            periods=days, freq='D')] # day - n_days <= dates < day    
    return df



def create_features(df, today, sequence_length):
    
    all_sequence = get_timespan(df, today, sequence_length).values
    
    group_store = all_sequence.reshape((-1, 10, sequence_length))
    
    store_corr = np.stack([np.corrcoef(i) for i in group_store], axis=0)
    
    store_features = np.stack([
              # Mean
              group_store.mean(axis=2),
              group_store[:,:,int(sequence_length/2):].mean(axis=2),
              # STD
              group_store.std(axis=2),
              group_store[:,:,int(sequence_length/2):].std(axis=2),
              # Skew
              skew(group_store, axis=2),
              # Kurtosis
              kurtosis(group_store, axis=2),
              np.apply_along_axis(lambda x: np.polyfit(np.arange(0, sequence_length), x, 1)[0], 2, group_store)
            ], axis=1)
    
    group_store = np.transpose(group_store, (0,2,1))
    store_features = np.transpose(store_features, (0,2,1))
    
    return group_store, store_corr, store_features

def create_label(df, today):
    
    y = df[today].values
    
    return y.reshape((-1, 10))