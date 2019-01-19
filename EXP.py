from MSE import MSE
# EMA: Exponential Smoothing / Exponential Moving Average
# Input
#   data: yahoo finance data in MAIN.py 
#   numDays: window size for moving average
#   db: boolean indicating whether double moving average
#   newData: boolean indicating whether we want to return transformed data set (alternative: MSE)
# Output
#   returns smoothed data set or mean squared error depending on 4th parameter
def EXP(data, numDays, db, newData):
    numData = len(data)
    if numData < numDays:
        raise ValueError('Data insufficient')
    exp = [0] * (numData - numDays + 1)
    exp[0] = sum(data[0 : numDays]) / numDays 
    mul = (2 / numDays + 1)
    for itr in range(numDays, numData):
        tdy = itr - numDays + 1
        exp[tdy] = (data[itr] - exp[tdy - 1]) * mul + exp[tdy - 1]
    if newData:
        return DEMA(exp, numDays) if db else exp
    return MSE(data[numDays * 2 - 2 : numData], DEMA(exp, numDays)) if db else MSE(data[numDays - 1 : numData], exp)

# DEMA: Double Exponential Moving Average
# Input
#   data: processed data from first EMA
#   numDays: window size for moving average
# Output
#   returns dataset after applying double EMA
def DEMA(data, numDays):
    numData = len(data)
    if numData < numDays:
        raise ValueError('Data insufficient')
    exp = [0] * (numData - numDays + 1)
    exp[0] = sum(data[0 : numDays]) / numDays 
    mul = (2 / numDays + 1)
    for itr in range(numDays, numData):
        tdy = itr - numDays + 1
        exp[tdy] = (data[itr] - exp[tdy - 1]) * mul + exp[tdy - 1]
    return [2 * data[idx + numDays - 1] - itr for idx, itr in enumerate(exp)]