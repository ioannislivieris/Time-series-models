# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Built-in libraries
#
import numpy  as np


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Scipy library
#
from   scipy.stats          import skew
from   scipy.stats          import kurtosis




def Descriptive_Statistics( Signal, verbose=True ):
    """
    Print desciptive statistics about a signal
    RMS
    Min
    Max
    Mean
    std
    Peak-2-Peak
    Zero-2-Peak
    Crest factor
    """
    
    RMS          = np.mean( Signal**2 )
    Max          = np.max( Signal )
    Min          = np.min( Signal )
    Mean         = np.mean( Signal )
    STD          = np.std( Signal )
    Median       = np.median( Signal )
    IQR          = np.percentile(Signal, 75) - np.percentile(Signal, 25)
    Skewness     = skew( Signal )
    Kurtosis     = kurtosis( Signal )
    #
    Peak_to_Peak = Max - Min
    Zero_to_Peak = max(map(abs, [Max, Min]))
    Crest_factor = Zero_to_Peak / RMS

    if (verbose == True):
        print('[INFO] Min:          %10.4f' % Min)
        print('[INFO] Max:          %10.4f' % Max)
        print('[INFO] Mean:         %10.4f' % Mean)
        print('[INFO] STD:          %10.4f' % STD)
        print('[INFO] Median:       %10.4f' % Median)
        print('[INFO] IQR           %10.4f' % IQR)
        print('[INFO] Skew:         %10.4f' % Skewness)
        print('[INFO] Kurtosis:     %10.4f' % Kurtosis)
        print()
        #
        #
        print('[INFO] RMS:          %10.4f' % RMS)
        print('[INFO] Peak to peak: %10.4f' % (Max - Min))
        print('[INFO] Zero to peak: %10.4f' % Zero_to_Peak)
        print('[INFO] Crest factor: %10.4f' % Crest_factor)
        print()


        print('\n')

    return (Min, Max, Mean, STD, Median, IQR, Skewness, Kurtosis, RMS, (Max - Min), Zero_to_Peak, Crest_factor)