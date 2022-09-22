import tensorflow.keras.backend as K
from   tensorflow.keras.losses import Loss

def SMAPE(y_true, y_pred):
    epsilon = 0.1
    summ = K.maximum(K.abs(y_true) + K.abs(y_pred) + epsilon, 0.5 + epsilon)
    smape = K.abs(y_pred - y_true) / summ * 2.0
    return smape
    
    
def MDA(y_true, y_pred):
    s = K.equal(K.sign(y_true[1:] - y_true[:-1]),
                 K.sign(y_pred[1:] - y_pred[:-1]))
    return K.mean(K.cast(s, K.floatx()))