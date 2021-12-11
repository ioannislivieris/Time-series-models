# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Built-in libraries
#
import time
import pandas as pd
import numpy  as np


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# FB-Prophet libraries
#
import fbprophet


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Sklearn library
#
from sklearn.metrics       import r2_score




def optimize_prophet_changepoint(data, initial_param, step, nIterations):
    '''
    This function is utilized for optimizing the Prophet model
    "changepoint_prior_scale" parameter should be optimized for maximum performance.
    Inputs:
        data: DataFrame
        initial_param: Initial value of changepoint parameter
        step: Increasement of changepoint value per Iteration
        nIterations: Number of Iterations
    '''
    R2          = {}
    changepoint = initial_param

    # Split training/validation set
    #
    index = int( 0.9*data.shape[0] )
    
    
    # The function iteratively changes the changepoint scale based 
    # on user-defined epoch range, measuring the error for each value. 

    for Iteration in range(nIterations):
        
        # Start timer
        #
        start = time.time()
        
        # Setup Prophet model
        #
        prophet = fbprophet.Prophet(daily_seasonality       = True, 
                                    yearly_seasonality      = False, 
                                    weekly_seasonality      = False,
                                    seasonality_mode        = 'multiplicative', 
                                    changepoint_prior_scale = changepoint)
        
        # Training model
        #
        prophet.fit(data[:index])
        
        # Stop timer
        #
        close = time.time()
        
        
        print('> Iteration: ', Iteration+1)
        print('[INFO] Prophet trained')
        print('[INFO] Time: {:.2f} seconds'.format(close-start))
        

        
        # Evaluation
        #
        # - Predictions
        forecast = prophet.predict( data )
        # - Calculate score (R2)
        score    = r2_score(data['y'][index:], forecast['yhat'][index:])
        # - Store results
        R2[changepoint] = score
        # 
        print('[INFO] Change-point: {:.2f}, R2 = {:.4f}\n\n'.format(changepoint, score) )

        
        # - Update parameters
        #
        changepoint += step        
        
        
        
        
    # The function automatically selects a changepoint scale parameter 
    # based on the lowest error rate.
    optim = max(list(R2.values()))
    for key,value in R2.items():
        if optim == value:
            k = key
            print('Optimal Changepoint is',key,'for an R2 of',value)

    return k




# # Setting the parameter to the optimal parameter identified by the optimization function.
# initial_changepoint = 0.5
# step                = 0.05
# Iterations          = 2

# changepoint = optimize_prophet_changepoint(df_train.reset_index(), initial_changepoint, step, Iterations)