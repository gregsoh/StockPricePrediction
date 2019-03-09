'''
This file consists of all the functions to evaluate how "good" our prediction is. 
'''
from DEBUG import MSE_CHECKSIZE
import math, statistics

'''
MSE: Mean Squared Error
Input
    dT: true values
    dP: predicted values
Output
    returns mean squared error (float)
'''
def MSE(dT, dP):
    MSE_CHECKSIZE(dT, dP)
    return sum([abs(dT[itr] - dP[itr]) for itr in range(len(dT))]) / len(dT)

'''
LMSE: Log Mean Squared Error
Input
    dT: true values
    dP: predicted values
Output
    returns log mean squared error (float)
'''
def LMSE(dT, dP):
    MSE_CHECKSIZE(dT, dP)
    return math.log(sum([abs(dT[itr] - dP[itr]) for itr in range(len(dT))]) / len(dT))

'''
DATAINFO: information of data
Input
    d: data
Output
    mean and standard deviation 
'''
def DATAINFO(d):
    return statistics.mean(d), statistics.stdev(d)