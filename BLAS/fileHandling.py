# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 17:21:04 2020

@author: poseidon
"""


import pandas as pd
import numpy  as np
from datetime import datetime

def ConvertDate( x ):
    year, month, day = x.split('-')
    
    return (pd.to_datetime(month + '/' + day + '/' + year))




def getData( filename, Strategy = 'Differencing', Horizon = 1 ):
    
    # Checking Strategy
    if (Strategy != 'Differenced' and Strategy != 'Returns'):
        print('[WARNING] Strategy: %s not known' % Strategy)
        Strategy = 'Differenced'
        
    print('[INFO] Strategy: %s was selected' % Strategy)
    
    
    
    
    Series = pd.read_csv( filename )
    Series = Series.interpolate()

    Series['Date'] = Series['Date'].apply(ConvertDate)
    Series.sort_values('Date')
    
    # Set date as index
    Series['Date'].astype('datetime64')
    Series.set_index('Date', inplace=True)

    
    if (Strategy == 'Differenced'):
        # Difference the Series        
        Transformed_Series = Series.diff()[1:]
        
        # m-lags
        for i in range(2, Horizon+1):
            Transformed_Series[ 'Diff_' + str(i) ] = Series.diff(i)
            
    else:
        Series = np.log( Series )

        # Difference the Series        
        Transformed_Series = Series.diff()[1:]
        
        for i in range(2, Horizon+1):
            Transformed_Series[ 'Diff_' + str(i) ] = Series.diff(i)
        
        #Transformed_Series = Smoothed_Series.diff()[1:] / Smoothed_Series[:-1].values
    
    
    Transformed_Series.dropna(inplace = True)
    
    return ( Series, Series.columns[0], Transformed_Series.dropna() )







def splitData(Series, Dates):
    valid_date_start = Dates[0]
    valid_date_end   = Dates[1]

    
    # Training Data
    Training   = Series[ Series.index < valid_date_start ]
    # Validation Data
    Validation = Series[ (Series.index >= valid_date_start) & (Series.index < valid_date_end)]
    # Testing Data
    Testing    = Series[Series.index >= valid_date_end ]

    return (Training, Validation, Testing)