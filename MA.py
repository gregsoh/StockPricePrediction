from MSE import MSE

'''
SMA: Simple Moving Average
Input
    data: processed data from first EMA
    numDays: window size for moving average
    newData: boolean indicating whether we want to return transformed data set (alternative: MSE)
Output
    returns dataset or MSE depending on third parameter
'''
def SMA(data, numDays, newData):
    numData = len(data)
    sim = [0] * (numData - numDays)
    for itr in range(numDays, numData):
        tdy = itr - numDays
        sim[tdy] = sum(data[itr - numDays : itr]) / numDays
    return sim if newData else MSE(data[numDays:], sim)

'''
EMA: Exponential Smoothing / Exponential Moving Average
Input
    data: yahoo finance data in MAIN.py 
    numDays: window size for moving average
    db: boolean indicating whether double moving average
    newData: boolean indicating whether we want to return transformed data set (alternative: MSE)
Output
    returns smoothed data set or mean squared error depending on 4th parameter
'''
def EMA(data, numDays, db, newData):
    numData = len(data)
    if numData < numDays:
        raise ValueError('Data insufficient')
    exp = [0] * (numData - numDays)
    exp[0] = sum(data[ : numDays]) / numDays
    alpha = 2 / (numDays + 1) #smoothing coefficient
    for itr in range(numDays + 1, numData):
        tdy = itr - numDays
        exp[tdy] = (data[itr - 1] - exp[tdy - 1]) * alpha + exp[tdy - 1]
    if newData:
        return DEMA(exp, numDays) if db else exp
    return MSE(data[numDays * 2:], DEMA(exp, numDays)) if db else MSE(data[numDays:], exp)
'''
DEMA: Double Exponential Moving Average
Input
    data: processed data from first EMA
    numDays: window size for moving average
Output
    returns dataset after applying double EMA
'''
def DEMA(data, numDays):
    numData = len(data)
    if numData < numDays:
        raise ValueError('Data insufficient')
    exp = [0] * (numData - numDays)
    exp[0] = sum(data[0 : numDays]) / numDays 
    alpha = 2 / (numDays + 1)
    for itr in range(numDays + 1, numData):
        tdy = itr - numDays
        exp[tdy] = (data[itr - 1] - exp[tdy - 1]) * alpha + exp[tdy - 1]
    return [2 * data[idx + numDays] - itr for idx, itr in enumerate(exp)]