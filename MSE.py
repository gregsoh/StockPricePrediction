'''
MSE: Mean Squared Error Calculation
Input
    dTrue: true values
    dPred: predicted values
Output
    returns mean squared error (float)
'''
def MSE(dTrue, dPred):
    if len(dTrue) != len(dPred):
        raise ValueError('Data size do not match')
    numData = len(dTrue)
    return sum([abs(dTrue[itr] - dPred[itr]) for itr in range(numData)]) / numData