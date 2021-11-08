# -*- coding: utf-8 -*-

import math
import pandas                           as pd
import numpy                            as np
import seaborn                          as sns
import matplotlib.pyplot                as plt 
#
from   sklearn                          import metrics
#
from   statsmodels.stats.diagnostic     import acorr_ljungbox
from   statsmodels.graphics.tsaplots    import plot_pacf
from   statsmodels.graphics.tsaplots    import plot_acf


def smape(A, F):
    return ( 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)) ) )




def RegressionEvaluation( Prices ):
    '''
    Parameters
    ----------
    Y : TYPE
        Real prices.
    Pred : TYPE
        Predicted prices.

    Returns
    -------
    MAE : TYPE
        Mean Absolute Error.
    RMSE : TYPE
        Root Mean Square Error.
    MAPE : TYPE
        Mean Absolute Percentage Error.
    R2   : TYPE
        R2 correlation
    '''
    
    SeriesName = Prices.columns[0]
    Prediction = Prices.columns[1]
    
    Y    = Prices[SeriesName].to_numpy()
    Pred = Prices[Prediction].to_numpy()
    
    
    
    MAE   = metrics.mean_absolute_error(Y, Pred)
    RMSE  = math.sqrt(metrics.mean_squared_error(Y, Pred))
    try:
        MAPE  = np.mean(np.abs((Y - Pred) / Y)) * 100.0
    except:
        MAPE  = np.NaN
        
    SMAPE = smape(Y, Pred)
    R2    = metrics.r2_score(Y, Pred)
    

        
    return (MAE, RMSE, MAPE, SMAPE, R2)




def ClassificationEvaluation( Prices, Horizon ):
    
    SeriesName = Prices.columns[0]
    Prediction = Prices.columns[1]
    
    Y    = []
    Pred = []
    for i in range(Horizon, Prices.shape[0]):
        print(Horizon, Prices[SeriesName][i-Horizon], Prices[SeriesName][i], Prices[Prediction][i])
        if (Prices[SeriesName][i] > Prices[SeriesName][i-Horizon]):
            Y.append(1)
        else:
            Y.append(0)
        
        if (Prices[Prediction][i] > Prices[SeriesName][i-Horizon]):
            Pred.append(1)
        else:
            Pred.append(0)
    
    Y    = np.array(Y)
    Pred = np.array(Pred)

    
    
    CM = metrics.confusion_matrix(Y, Pred)
    
    Accuracy = metrics.accuracy_score(Y, Pred)
    F1       = metrics.f1_score(Y, Pred)
    
    fpr, tpr, thresholds = metrics.roc_curve(Y, Pred)
    AUC = metrics.auc(fpr, tpr)

    Sen      = CM[1][1] / (CM[1][1] + CM[1][0])
    Spe      = CM[0][0] / (CM[0][0] + CM[0][1])    

    
    if (sum(CM[:][1]) != 0):
        PPV      = CM[1][1] / (CM[1][1] + CM[0][1])
    else:
        PPV      = 0.0
        
    if ( (CM[0][0] + CM[1][0]) != 0):
        NPV      = CM[0][0] / (CM[0][0] + CM[1][0])
    else:
        NPV      = 0.0

    
    GM = math.sqrt( CM[0][0] * CM[1][1] )
    
    
    return (CM, Accuracy, AUC, F1, GM, Sen, Spe, PPV, NPV)


def AutoCorrelationTest(Y, Pred):   
    # Auto-corretion check (by Stavroyiannis)
    D = Y - Pred

    # ACF plot
    sns.set( rc = {'figure.figsize':(15, 2)})
    
    fig = plot_acf(D, lags=5)
    fig.set_dpi(100)
    fig.set_figheight(3.0)
        
#    # PCF plot
#    fig = plot_pacf(D, lags=5)
#    fig.set_dpi(100)
#    fig.set_figheight(3.0)
#    plt.show()           
        
    # Apply
    # 1. Ljung-Box Q-test
    # 2. Box-Pierce test    
    TestForResidualCorrelation(D) 
    
    
    
def TestForResidualCorrelation(D):

    lbvalue, pvalue, bpvalue, bppvalue = acorr_ljungbox(D, lags=[10], boxpierce=True)    

    print('Ljung-Box Q-test    ***   p-value = %5.4f' % pvalue)
    if (pvalue > 0.05):
        print('H0 is ACCEPTED\n\n')
    else:
        print('H0 is Rejected\n\n')
    
    print('Box-Pierce test     ***   p-value = %5.4f' % bppvalue)
    if (bppvalue > 0.05):
        print('H0 is ACCEPTED\n\n')
    else:
        print('H0 is Rejected\n\n')


def ConfusionMatrixVisualize( CM ):
    #get pandas dataframe
    df_cm = pd.DataFrame(CM, columns=['Down', 'Up'])
    
    df_cm['Real'] = ['Down','Up']
    df_cm = df_cm.set_index('Real')
    
    #colormap: see this and choose your more dear

    
    sns.set( rc = {'figure.figsize':(4,4)})
    sns.set(font_scale=1.5) # for label size
    sns.heatmap(df_cm, annot=True, fmt='d', annot_kws={"size": 16}) # font size
    plt.show()
#
