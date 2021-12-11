# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Built-in libraries
#
import pandas as pd


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Sklearn library
#
from sklearn.linear_model import LinearRegression



def interpretationVIF(x):
    if (x == 1):
        return ('Not collerated')
    elif (x < 5):
        return ('Moderately correlated')
    elif (x < 10):
        return ('Highly correlated')
    else:
        return ('Extremely highly correlated')




def calculateVIF(df, features):   
    '''
    1       — features are not correlated
    1<VIF<5 — features are moderately correlated
    VIF>5   — features are highly correlated
    VIF>10  — high correlation between features and is cause for concern
    '''
    
    # All the features that you want to examine
    #
    vif, tolerance = {}, {}    
    
    
    for feature in features:
        # Extract all the other features you will regress against
        #
        X = [f for f in features if f != feature]        
        
        # Extract r-squared from the fit
        #
        X, y = df[X], df[feature]     
        
        # Calculate R2
        #
        r2 = LinearRegression().fit(X, y).score(X, y)                
        
        # Calculate tolerance
        #
        tolerance[feature] = 1 - r2        
        
        # Calculate VIF
        # 
        vif[feature] = 1 / tolerance[feature]    

    # Return VIF DataFrame
    #
    R = pd.DataFrame({'VIF': vif, 'Tolerance': tolerance}).sort_values('Tolerance')
    
    # VIF interpretation
    #
    R['Status'] = R['VIF'].apply( interpretationVIF )
    
    return (R)