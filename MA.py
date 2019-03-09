'''
This file consists of different moving averages algorithms. 
'''
from MEASURE import MSE, LMSE

'''
SMA: Simple Moving Average
Input
    d: yahoo finance data in MAIN.py 
    w: window size for moving average
    rv: when rv = True, return transformed data set (else return MSE)
    er: error fn (0: MSE; 1: log MSE)
Output
    returns dataset or MSE depending on third param
'''
def SMA(d, w, rv, er = 0):
    num = len(d)
    sim = [d[0] if itr == 0 else sum(d[ : itr]) / len(d[ : itr]) if itr <= w else sum(d[itr - w : itr]) / len(d[itr - w : itr]) for itr in range(num)]
    return sim if rv else MSE(d, sim) if er == 0 else LMSE(d, sim)

'''
EMA: Exponential Moving Average
Input
    d: yahoo finance data in MAIN.py 
    w: window size for moving average
    a: smoothing coefficient
    db: boolean indicating whether double moving average
    rv: when rv = True, return transformed data set (else return MSE)
    er: error fn (0: MSE; 1: log MSE)
Output
    returns smoothed data set or MSE depending on 4th param
'''
def EMA(d, w, a, db, rv, er = 0):
    num = len(d)
    exp = [d[itr] if itr == 0 else sum(d[ : itr]) / len(d[ : itr]) if itr <= w else 0 for itr in range(num)]
    for itr in range(w + 1, num):
        exp[itr] = (d[itr - 1] - exp[itr - 1]) * a + exp[itr - 1]
    if db:
        return DEMA(exp, w, a) if rv else MSE(d, DEMA(exp, w, a)) if er == 0 else LMSE(d, DEMA(exp, w, a))
    return exp if rv else MSE(d, exp) if er == 0 else LMSE(d, exp)

'''
DEMA: Double Exponential Moving Average
Input
    d: processed data from first EMA
    w: window size for moving average from EMA
    a: smoothing coefficient
Output
    returns transformed dataset
'''
def DEMA(d, w, a):
    num = len(d)
    exp = [d[itr] if itr == 0 else sum(d[ : itr]) / len(d[ : itr]) if itr <= w else 0 for itr in range(num)] 
    for itr in range(w + 1, num):
        exp[itr] = (d[itr - 1] - exp[itr - 1]) * a + exp[itr - 1]
    return [2 * d[idx] - itr for idx, itr in enumerate(exp)]