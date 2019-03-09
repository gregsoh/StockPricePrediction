import numpy as np
import math
'''
This debug file does sanity check for functions in other files. 
Function name is fileName + "_" + description of check
'''


'''
GENERAL_CHECKSIZE: checks that two lists have the same size
Input
    l1 and l2: two lists
    err: specific error
No return value
'''
def GENERAL_CHECKSIZE(l1, l2, err):
    if len(l1) != len(l2):
        raise ValueError('General error: list sizes must be identical ' + err)

'''
GENERAL_CHECKEMPTY: Checks that the list is not empty
Input
    d: list/ data
    err: specific error
No return value
'''
def GENERAL_CHECKEMPTY(d, err):
    if len(d) == 0:
        raise ValueError('General error: list is empty ' + err)

'''
GENERAL_CHECKVAL: Checks that value is between a range
Input
    v: value 
    l: lower bound
    u: upper bound
    err: specific error
'''
def GENERAL_CHECKVAL(v, l, u, err):
    if v < l or v > u:
        raise ValueError('General error: value is out of bounds ' + err)
'''
MAIN_DOWNLOAD: checks that all data is downloaded successfully
Input
    dataset: list of datasets
'''
def MAIN_DOWNLOAD(dataset):
    for itr in (dataset):
        if len(itr) <= 0:
            raise ValueError('MAIN.py: data downloading issues')

'''
MAIN_CHECKDATA: check that data has been processed correctly by checking type and equates -0,0 with 0.0
Input
    d: dataset
No return value
'''
def MAIN_CHECKDATA(d):
    for idx, row in d.iterrows():
        for colName, itm in row.iteritems():
            if type(itm) != np.float64 and type(itm) != float and type(itm) != int:
                raise ValueError('DEBUG.py: Data type is not a float.')
            if type(itm) == np.ndarray:
                raise ValueError('DEBUG.py: Data type cannot be ndarray.')
            if itm == -0.0: #removes all negative 0
                d.at[idx, colName] = 0.0

'''
MSE_CHECKSIZE: checks for equality of size between two datasets
Input
    d1: data1
    d2: data2
No return value
'''
def MSE_CHECKSIZE(d1, d2):
    if len(d1) != len(d2):
        raise ValueError('MSE.py: data sizes do not match to evaluate MSE')

'''
DT_DATAENOUGH: check if there is enough data
Input
    d: data
    l: required length
No return value
'''
def DT_DATAENOUGH(d, l):
    if len(d) < l:
        raise ValueError('DT.py: Inadequate data to create the feature.')


'''
DT_CHECKDATA: check that data makes sense
Input
    r: result from classification/ regression
    b: boolean if it is binary
No return value
'''
def DT_CHECKVALUES(r, binary):
    if len(r) == 0:
        raise ValueError('DT.py: The data set is empty which is not supposed to be the case.')
    possible = set([-1, 1])
    for itr in r:
        if binary: 
            if itr not in possible:
                raise ValueError('DT.py: This is binary classification. It has to be -1 or 1')

'''
LSTM_CHECK: check for LSTM
Input
    data: the data we want to parse
    iterationNum: to see which iteration is this
    add1, add2, add3, add4: additional printing that we want to print
No return value 
'''
def LSTM_CHECK(d, iterationNum, add1 = None, add2 = None, add3 = None, add4 = None):
    for itr in d:
        for itr2 in itr:
            var = list(itr2)
            for itr3 in var:
                if itr3 != itr3: #When it does not equal to itself, then it is not a number
                    if add1 != None:
                        print(add1)
                    if add2 != None:
                        print(add2)
                    if add3 != None:
                        print(add3)
                    if add4 != None:
                        print(add4)
                    # print("Printing each element in output:", var)
                    raise ValueError('LSTM.py: Something wrong with value in iteration ' + str(iterationNum))

'''
LSTM_NAN: check if a list has nan
Input
    d: list
'''
def LSTM_NAN(d):
    for itr in d:
        if math.isnan(itr):
            raise ValueError('LSTM.py: There is nan in the list')
'''
LSTM_CHECKDIM: check processed data dimensions
Input
    data: the data we want to parse
    numElem: number of element in data
    B: batchsize
    D: dimensions
No return value 
'''
def LSTM_CHECKDIM(d, numElem, B, D):
    if len(d) != numElem:
        print(len(d), numElem)
        raise ValueError('LSTMCHECK: length of data incorrect')
    for itr in d:
        if itr.shape != (B, D): 
            raise ValueError('LSTM.py: dimension is wrong; probably not converted to onehot')