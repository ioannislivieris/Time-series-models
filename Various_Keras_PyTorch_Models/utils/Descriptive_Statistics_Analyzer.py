# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Built-in libraries
#
import numpy  as np
import pandas as pd

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Scipy library
#
from   scipy.stats          import skew
from   scipy.stats          import kurtosis




class DescriptiveStatisticsAnalyzer():
    def __init__(self, data = None, rule = '1W'):
        '''
        Inputs:
            data:    Import data (DataFrame)
            feature: Name of requested feature (str)
            step:    The offset string or object representing target conversion (default: 1W)
        '''
        self.data    = data
        self.rule    = rule
        
 


    def showAvailableStatistics( self ):
        '''
            Show Available Statistics
        '''
        
        print('[INFO] Available Statistics')
        print('- Min')
        print('- Max')
        print('- Mean')
        print('- STD (standard deviation)')
        print('- Median')
        print('- IQR (Interquartile range)')
        print('- Skewness')
        print('- Kurtosis')
        print('- RMS')
        print('- MaxMin')
        print('- Zero_to_Peak')
        print('- Crest_factor')
        


    def createStatistics( self ):
        '''
            Create descriptive stastistics for all features
        '''        
        
        if (self.data is None):
            print('[ERROR] No dataframe was provided')
            return

        
        # Get rule for resampling
        #
        rule = self.rule
        
        
        # Get descriptive statistics on resampled data
        #
        Stats = {
                'Min':          self.data.resample( rule ).apply(lambda x: np.min(x)),
                'Max':          self.data.resample( rule ).apply(lambda x: np.max(x)),     
                'Mean':         self.data.resample( rule ).apply(lambda x: np.mean(x)),
                'STD':          self.data.resample( rule ).apply(lambda x: np.std(x)),
                'Median':       self.data.resample( rule ).apply(lambda x: np.median(x)),
                'IQR':          self.data.resample( rule ).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25)),
                #
                'Skewness':     self.data.resample( rule ).apply(lambda x: skew(x)),
                'Kurtosis':     self.data.resample( rule ).apply(lambda x: kurtosis(x)),
                # 
                'RMS':          self.data.resample( rule ).apply(lambda x: np.mean(x**2)),
                'MaxMin':       self.data.resample( rule ).apply(lambda x: np.max(x) - np.min(x)),
                'Zero_to_Peak': self.data.resample( rule ).apply(lambda x: np.max( np.abs(x) ) ),
                'Crest_factor': self.data.resample( rule ).apply(lambda x: np.max( np.abs(x) ) / np.mean(x**2))

            }
        
        self.Stats = Stats
        
        
    def getDescriptiveStatistics(self, metric = 'Mean'):
        '''
            Retrieve descriptive stastistics for all features
            Inputs:
                metric: Requested metric e.g.: Min, Max, Mean, etc
        '''
        return ( d[metric] )
    
    def visualize(self, feature=None, metric='RMS'):
        
        if (self.data is None):
            print('[ERROR] No dataframe was provided')
            return


        if (feature is None):
            
            for feature in self.data.columns:
                try:
                    self.Stats[ metric ][feature].plot(legend=True, figsize=(15, 3) )
                    plt.title( 'Metric: {} (Rule: {})'.format(metric, self.rule) )
                    plt.show()
                except Exception as e: 
                    print('[ERROR] Exception: '.format(e.__class__.__name__) )
                    print(' > ', feature)

        else:
            try:
                self.Stats[ metric ][ feature ].plot(legend=True, figsize=(15, 5) )
                plt.title( 'Metric: {} (Rule: {})'.format(metric, self.rule) )
                plt.show()
            except Exception as e: 
                print('[ERROR] Exception: '.format(e.__class__.__name__) )
                print(' > ', feature)


    
    