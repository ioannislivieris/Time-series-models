# Libraries
import pandas as pd

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss




class StationarityTests:
    def __init__(self, significance = .05):
        self.SignificanceLevel = significance
        self.pValue            = None
        self.isStationary      = None     
        
        
    def ADF(self, timeseries, verbose = True):
        # Augmented Dickey-Fuller test
        adfTest = adfuller(timeseries, autolag = 'BIC')
        # p-value
        pValue = adfTest[1]
            
       
        # Test results
        dfResults = pd.Series(adfTest[0:4], index=['ADF Test Statistic', 'p-value', '# Lags Used', '# Observations Used'])
               
                    
        # Add Critical Values
        for key,value in adfTest[4].items():
            dfResults['Critical Value (%s)'%key] = value
            
            
            
            
        if (verbose == True):   
            print('* Augmented Dickey-Fuller test *')
            print(dfResults)
        
            # Stationarity check 
            if (pValue < self.SignificanceLevel):
                print('[INFO] Series is stationary\n\n')                
            else:
                print('[INFO] Series is non-stationary\n\n')

        
      
        return (dfResults[0], dfResults[5], dfResults[1]) 
    
    
    
    
    
    
    def KPSS(self, timeseries, verbose = True):
        # Kwiatkowski–Phillips–Schmidt–Shin test
        kpsstest = kpss(timeseries, regression='c')
        # p-value
        pValue = kpsstest[1]
        
        
        # Test results
        kpss_output = pd.Series(kpsstest[0:3], index=['KPSS test statistic', 'p-value', '# Lags Used'])        
        
        
        for key, value in kpsstest[3].items():
            kpss_output['Critical Value (%s)' % key] = value
            
            
            
        if (verbose == True):
            print('* Kwiatkowski–Phillips–Schmidt–Shin test *')
            print(kpss_output)
                        
            # Stationarity check    
            if (pValue < self.SignificanceLevel):
                print('[INFO] Series in non-stationary\n\n')
            else:
                print('[INFO] Series is stationary\n\n')            

            
        return(kpss_output[0], kpss_output[3], kpss_output[1])