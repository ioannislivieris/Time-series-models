import warnings
warnings.filterwarnings('ignore')


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Basic libraries
#
import os
import numpy as np
import pandas as pd
import os

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# PyTorch libraries
#
import torch
from   torch.utils.data      import Dataset, DataLoader


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# User library
#
from   utils.timefeatures    import time_features


class createDataset(Dataset):
    def __init__(self, df, size=None, features='S', target=None, timeenc=1, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len   = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len  = 24 * 4
        else:
            self.seq_len   = size[0]
            self.label_len = size[1]
            self.pred_len  = size[2]
        
        if (target is None):
            target = df.columns[-1]

        self.df       = df.reset_index()
        self.features = features
        self.target   = target
        self.timeenc  = timeenc
        self.freq     = freq

        self.__read_data__()

    def __read_data__(self):
        
        df_raw = self.df
        
        # Re-arrange columms
        #
        # Example: df_raw.columns: ['date', ...(other features), target feature]
        #
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        
        print('[INFO] Features: ', cols)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data   = df_raw[cols_data]
        elif self.features == 'S':
            df_data   = df_raw[[self.target]]


        # Get data
        #
        data = df_data.values
            
            
        # Date/Time features
        #
        df_stamp         = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        
        if self.timeenc == 0:
            df_stamp['month']   = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day']     = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour']    = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
            
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        # Create data
        #
        self.data_x     = data
        self.data_y     = data
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end   = s_begin + self.seq_len
        
        r_begin = s_end   - self.label_len
        r_end   = r_begin + self.label_len + self.pred_len

        seq_x      = self.data_x[s_begin:s_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]

        seq_y      = self.data_y[r_begin:r_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1




