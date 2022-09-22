# Built-in libraries
import math
import numpy as np

# Sklearn
#
from sklearn                 import metrics





# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# =              Regression metrics             =
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def smape(A, F):
    return ( 100.0/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)) ) )


def mape(A, F):
    return( np.mean(np.abs((A - F) / A)) * 100.0 )


def rmse(A, F):
    return ( math.sqrt(metrics.mean_squared_error(A, F)) )


def mae(A, F):
    return ( metrics.mean_absolute_error(A, F) )


def R2(A, F):
    return ( metrics.r2_score(A, F) )