from MA import EMA, SMA
import numpy as np
'''
This truncating constant is tobin the values so that the decision tree will not have too many child nodes (Most values are rounded to the nearest tens place)
'''
TRUNCONST = 10

'''
RSI: Relative Strength Index
Input
    data: yahoo finance data in MAIN.py 
Output
    returns RSI data set
'''
def RSI(data):
    numData = len(data)
    if numData < 15:
        raise ValueError('Data insufficient')
    gainOrLoss, RS = [0] * (numData - 1), [0] * (numData - 1 - 14)
    for itr in range(1, numData):
        gainOrLoss[itr - 1] = data[itr] - data[itr - 1]
    for itr in range(14, numData - 1):
        gain, loss = [], []
        for idx in range(14):
            gain.append(gainOrLoss[itr - idx - 1]) if gainOrLoss[itr - idx - 1] > 0 else loss.append(gainOrLoss[itr - idx - 1])
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
    if numData < 14:
        raise ValueError('Data insufficient')
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
    winOne: the shorter window; winTwo: the longer window 
Output
    returns data set with MACD values
'''
def MACD(data, winOne, winTwo):
    MA_1, MA_2 = EMA(data, winOne, False, True), EMA(data, winTwo, False, True)
    return [(((x - y) // TRUNCONST) * TRUNCONST) for x, y in zip(MA_1[winTwo - winOne : ], MA_2)]

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
            raise ValueError('OBV type has an issue')
    return OBVData

'''
FEATURE: features for training
Input
    data: data set
Output
    returns the processed dataset or PD
'''
def FEATURE(data):
    winShort, winLong = 12, 26
    F1 = RSI(data.Close) # Feature 1: Relative Strength Index
    F2 = SO(data, False) # Featue 2: Stochastic Oscillator
    F3 = SO(data, True) # Feature 3: Williams %R, related to feature 2 
    F4 = MACD(data.Close, winShort, winLong) # Feature 4: Moving average convergence divergence
    F5 = PRC(data.Close, 12) # Feature 5: Price Rate of Change
    F6 = OBV(data) # Feature 6: On-balance Volume
    PD = data[winLong : ] 
    PD = PD.assign(RSI = F1[winLong - 15 : ])
    PD = PD.assign(SO = F2[winLong - 13 : ])
    PD = PD.assign(WIL = F3[winLong - 13 : ])
    PD = PD.assign(MACD = F4)
    PD = PD.assign(PRC = F5[winLong - winShort : ])
    PD = PD.assign(OBV = F6[winLong : ])
    return PD