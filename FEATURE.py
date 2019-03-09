'''
This python file codes the features used in the random forests. They are
    |- Relative Strength Index
    |- Stochastic Oscillator
    |- Moving Average Convergence Divergence
    |- Price Rate of Change
    |- On-balance volume
All these functions are called by FEATURE method in this code base
'''
from MA import EMA, SMA
from DEBUG import DT_DATAENOUGH
import numpy as np

TRUNCONST = 10 #This truncating constant is to bin the values so that the decision tree will not have too many child nodes (Most values are rounded to the nearest "tens" place)

'''
RSI: Relative Strength Index
Input
    data: yahoo finance data in MAIN.py 
Output
    returns RSI data set
'''
def RSI(data):
    numData = len(data)
    DT_DATAENOUGH(data, 15)
    GL, RS = [0] * (numData - 1), [0] * (numData - 1 - 14) #GL is gain or loss
    for itr in range(1, numData):
        GL[itr - 1] = data[itr] - data[itr - 1]
    for itr in range(14, numData - 1):
        gain, loss = [], []
        for idx in range(14):
            gain.append(GL[itr - idx - 1]) if GL[itr - idx - 1] > 0 else loss.append(GL[itr - idx - 1])
        RS[itr - 14] = (sum(gain) / len(gain)) / (-sum(loss) / len(loss))
    RSI = [((100 - 100 / (1 + itr)) // TRUNCONST) * TRUNCONST for itr in RS]
    return RSI

'''
SO: Stochastic Oscillator (with a boolean option to calculate the williams %R)
Input
    data: yahoo finance data in MAIN.py 
    Wil: Do we want to calculate the closely related Williams %R? 
Output
    returns data set with SO values
'''
def SO(data, wil):
    numData = len(data.Close)
    DT_DATAENOUGH(data, 14)
    RVAL = [0] * (numData - 14 + 1)
    for itr in range(13, numData):
        trackLow, trackHigh = data.Low[itr - 13 : itr + 1], data.High[itr - 13 : itr + 1]
        if wil: 
            RVAL[itr - 13] = ((-100 * (max(trackHigh) - data.Close[itr]) / (max(trackHigh) - min(trackLow))) // TRUNCONST) * TRUNCONST
        else:
            RVAL[itr - 13] = ((100 * (data.Close[itr] - min(trackLow)) / (max(trackHigh) - min(trackLow)))  // TRUNCONST) * TRUNCONST
    return RVAL

'''
MACD: Moving Average Convergence Divergence
Input
    data: yahoo finance data in MAIN.py 
    w_short: the shorter window; w_long: the longer window 
Output
    returns data set with MACD values
'''
def MACD(data, w_short, w_long):
    MA_1, MA_2 = EMA(data, w_short, 2 / (w_short + 1), False, True), EMA(data, w_long, 2 / (w_long + 1), False, True)
    return [(((x - y) // TRUNCONST) * TRUNCONST) for x, y in zip(MA_1[w_long - w_short : ], MA_2)]

'''
PRC: Price Rate of Change
Input
    data: yahoo finance data in MAIN.py 
    win: window
Output
    returns data set with PRC values
'''
def PRC(data, win):
    numData = len(data)
    PRCData = [0] * (numData - win)
    for itr in range(len(PRCData)):
        PRCData[itr] = round((data[itr + win] - data[itr]) / data[itr], 2)
        PRCData[itr] = round(PRCData[itr] * 100) / 100
    return PRCData

'''
OBV: On-balance volume
Input
    data: data set
Output
    returns the processed dataset or OBV
'''
def OBV(data):
    DIVCONST = 1000000000
    numData = len(data.Close)
    OBVData = [0] * numData
    OBVData[0] = 0.
    for itr in range(1, numData):
        if data.Close[itr] > data.Close[itr - 1]: 
            OBVData[itr] = OBVData[itr - 1] + (data.Volume[itr] // DIVCONST) * DIVCONST
        elif data.Close[itr] < data.Close[itr - 1]:
            OBVData[itr] = OBVData[itr - 1] - (data.Volume[itr] // DIVCONST) * DIVCONST
        else:
            OBVData[itr] = OBVData[itr - 1]
        if type(OBVData[itr]) != float and type(OBVData[itr]) != np.float64:
            raise ValueError('FEATURE.py: OBV has an issue')
    return OBVData

'''
FEATURE: features for training
Input
    data: data set
Output
    returns the processed dataset or PD
'''
def FEATURE(data):
    w_Short, w_Long, numDays = 12, 26, 12
    F1 = RSI(data.Close) # Feature 1: Relative Strength Index
    F2 = SO(data, False) # Featue 2: Stochastic Oscillator
    F3 = SO(data, True) # Feature 3: Williams %R, related to feature 2 
    F4 = MACD(data.Close, w_Short, w_Long) # Feature 4: Moving average convergence divergence
    F5 = PRC(data.Close, numDays) # Feature 5: Price Rate of Change
    F6 = OBV(data) # Feature 6: On-balance Volume
    PD = data[w_Long : ] 
    # print(len(F1[w_Long - 15 : ]), len(F2[w_Long - 13 : ]), len(F3[w_Long - 13 : ]), len(F4[w_Short : ]), len(F5[w_Long - numDays : ]), len(F6[w_Long : ]))
    PD = PD.assign(RSI = F1[w_Long - 15 : ])
    PD = PD.assign(SO = F2[w_Long - 13 : ])
    PD = PD.assign(WIL = F3[w_Long - 13 : ])
    PD = PD.assign(MACD = F4[w_Short : ])
    PD = PD.assign(PRC = F5[w_Long - numDays : ])
    PD = PD.assign(OBV = F6[w_Long : ])
    return PD