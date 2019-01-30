import numpy as np
'''
CHECKDATA: check that data has been processed correctly
Input
    d: dataset
Output
    no return value
'''
def CHECKDATA(d):
    for idx, row in d.iterrows():
        for colName, itm in row.iteritems():
            if type(itm) != np.float64 and type(itm) != float and type(itm) != int:
                raise ValueError('DEBUG.py: Data type is not a float.')
            if type(itm) == np.ndarray:
                raise ValueError('DEBUG.py: Data type cannot be ndarray.')
            if itm == -0.0: #removes all negative 0
                d.at[idx, colName] = 0.0

'''
CHECKDATA: check that data makes sense
Input
    r: result from classification/ regression
    b: boolean if it is binary
Output
    no return value
'''
def CHECKVALUES(r, binary):
    possible = set([-1, 1])
    for itr in r:
        if binary: 
            if itr not in possible:
                raise ValueError('This is binary classification. It has to be -1 or 1')