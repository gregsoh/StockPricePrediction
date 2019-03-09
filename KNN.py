'''
This file consists three types of KNN:
    |- Normal KNN
    |- Weighted KNN
    |- Modified KNN 
'''
from MEASURE import MSE
from DEBUG import GENERAL_CHECKSIZE
from statistics import mode
import matplotlib.pyplot as plt, matplotlib.dates as mdates, numpy as np

'''
KNN: K nearest neighbor
Input
    train: training dataset
    test: testing dataset
    maxMin: dictionary keeping track of max and min for each feature so that we can calculate for the euclidean distance on unseen data set
    k: paramter for k nearest neighbor
    forecast: forecast whether price rise after k days (set to 1); forecast prices after k days (set to 2)
    weightedKNN: boolean to indicate if we want weighted
    modifiedKNN: bool to indicate if we want modified version of KNN
Ouput
    return MSE, predicted
'''
def KNN(train, test, maxMin, k, forecast, weightedKNN, modifiedKNN):
    def weights(k):
        denom = (k + 1) * k / 2
        return [itr / denom for itr in range(k, 0, -1)]

    def ED(testData, procD, k, forecast, weights = None):
        distTrack, classTrack = [], []
        for idx, itr in enumerate(procD):
            arr, val = itr[0], itr[1]
            distTrack.append(sum([(a - b) ** 2.0 for a, b in zip(testData, arr)]))
        idx = np.argpartition(np.array(distTrack), k)
        for itr in idx[ : k]:
            classTrack.append(procD[itr][1])
        if weights == None:
            return sum(classTrack) / k if forecast == 2 else mode(classTrack)
        else:
            avg = sum([a * b for a, b in zip(classTrack, weights)])
            return avg if forecast == 2 else 1 if avg > 0 else -1

    if forecast == 2 and modifiedKNN:
        raise ValueError('KNN.py: forecast parameter cannot be 2 when we we use MKNN')

    if modifiedKNN: 
        return MKNN(train, test, maxMin, k)

    PD = processData(train)
    predicted = []
    for idx, row in test.iterrows():
        feature = [(itm - maxMin[CN][0]) / (maxMin[CN][1] - maxMin[CN][0]) for CN, itm in row.iteritems() if CN != 'Target']
        predicted.append(ED(feature, PD, k, forecast, weights(k))) if weightedKNN else predicted.append(ED(feature, PD, k, forecast))
    return MSE(test['Target'], predicted), predicted

'''
MKNN: Modified K nearest neighbor
Input
    train: training dataset
    test: testing dataset
    maxMin: dictionary keeping track of max and min for each feature so that we can calculate for the euclidean distance on unseen data set
    k: paramter for k nearest neighbor
Output
    return MSE, predictedData
Based on http://www.iaeng.org/publication/WCECS2008/WCECS2008_pp831-834.pdf
'''
def MKNN(train, test, maxMin, k):
    def validity(d, k):
        valid = []
        for idx, itr in enumerate(d):
            feature, val = itr[0], itr[1]
            distTrack, ttl = [], 0
            for idx2, itr2 in enumerate(d):
                arr, val2 = itr2[0], itr2[1]
                distTrack.append(sum([(a - b) ** 2.0 for a, b in zip(feature, arr)]))
            track = np.argpartition(np.array(distTrack), k + 1)
            for elem in track[ : k + 1]:
                ttl += 1 if d[elem][1] == val else 0
            if ttl - 1 < 0:
                raise ValueError('KNN.py: this value should always be non negative')
            valid.append((ttl - 1.) / float(k))
        return valid
    def MED(testD, procD, k, v): #stands for modified euclidean distance
        distTrack, classTrack, validityTrack, weights = [], [], [], []
        for idx, itr in enumerate(procD):
            GENERAL_CHECKSIZE(itr[0], testD, '(KNN.py: train data point and test data point)')
            arr, val = itr[0], itr[1]
            distTrack.append(sum([(a - b) ** 2.0 for a, b in zip(testD, arr)]))
        track = np.argpartition(np.array(distTrack), k)
        for itr in track[ : k]:
            classTrack.append(procD[itr][1])
            validityTrack.append(v[itr])
        GENERAL_CHECKSIZE(validityTrack, classTrack,'(KNN.py: validty and classTrack)'), GENERAL_CHECKSIZE(classTrack, track[ : k], '(KNN.py: validty and track[ : k])')
        for idx, itr in enumerate(validityTrack):
            weights.append(itr * 1 / (distTrack[track[idx]] + 0.5))
        avg = sum([a * b for a, b in zip(classTrack, weights)]) 
        return 1 if avg > 0 else -1
    PD = processData(train)
    valid = validity(PD, k)
    predicted = []
    for idx, row in test.iterrows():
        feature = [(itm - maxMin[CN][0]) / (maxMin[CN][1] - maxMin[CN][0]) for CN, itm in row.iteritems() if CN != 'Target']
        predicted.append(MED(feature, PD, k, valid))
    return MSE(test['Target'], predicted), predicted

'''
processData: processing data
Input
    d: dataset
Ouput
    returns arrary of tuples; first element is feature vector and the second element is the target value
'''
def processData(d):
    return [([itm for colName, itm in row.iteritems() if colName != 'Target'],  row['Target']) for idx, row in d.iterrows()]