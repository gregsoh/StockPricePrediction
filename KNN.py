from MSE import MSE
import numpy as np
from statistics import mode
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
'''
KNN: K nearest neighbor
Input
    train: training dataset
    test: testing dataset
    k: paramter for k nearest neighbor
    graph: do we want to plot a graph
    forecastType: forecast whether price rise after k days (set to 1); forecast prices after k days (set to 2)
    weightedKNN: boolean to indicate if we want weighted
    ModifiedKNN: bool to indicate if we want modified version of KNN
Ouput
    return MSE
'''
def KNN(train, test, k, graph, forecastType, weightedKNN, ModifiedKNN):
    def weights(k):
        denom = (k + 1) * k / 2
        return [itr / denom for itr in range(k, 0, - 1)]
    if forecastType == 2 and ModifiedKNN:
        raise ValueError('KNN.py: forecastType cannot be 2 when we use modifiedKNN') 
    if ModifiedKNN: 
        return MKNN(train, test, k, graph)   
    arr = pData(train)
    predicted = []
    for idx, row in test.iterrows():
        li = [itm for colName, itm in row.iteritems() if colName != 'Target' or colName != 'Volume']
        predicted.append(ED(li, arr, k, forecastType, weights(k))) if weightedKNN else predicted.append(ED(li, arr, k, forecastType))
    if graph:
        yAxisLabel = 'Decision' if forecastType == 1 else 'Prices'
        var = 1 if weightedKNN else 0 #var stands for variable
        drawGraph(test['Target'], predicted, yAxisLabel, var)
    return MSE(test['Target'], predicted)

'''
MKNN: Modified K nearest neighbor
Input
    train: training dataset
    test: testing dataset
    k: paramter for k nearest neighbor
    graph: do we want to plot a graph
Ouput
    return MSE
'''
def MKNN(train, test, k, graph):
    def validity(train, k):
        valid = []
        for idx, itr in enumerate(train):
            feature, val = itr[0], itr[1]
            distTrack, ttl = [], 0
            for idx2, itr2 in enumerate(train):
                arr, val2 = itr2[0], itr2[1]
                distTrack.append(sum([(a - b) ** 2.0 for a, b in zip(feature, arr)]))
            track = np.argpartition(np.array(distTrack), k + 1)
            for elem in track[ : k + 1]:
                ttl += 1 if train[elem][1] == val else 0
            if ttl - 1 < 0:
                raise ValueError('KNN.py: this value should always be non negative')
            valid.append((ttl - 1.) / float(k))
        return valid
    arr = pData(train)
    valid = validity(arr, k)
    predicted = []
    for idx, row in test.iterrows():
        li = [itm for colName, itm in row.iteritems() if colName != 'Target' or colName != 'Volume']
        predicted.append(MED(li, arr, k, valid))
    if graph:
        yAxisLabel = 'Decision'
        drawGraph(test['Target'], predicted, yAxisLabel, 2)
    return MSE(test['Target'], predicted)

'''
drawGraph: draw the graph of predicted versus actual
Input
    trueVal: true value
    predVal: predicated value
    yAxisLabel: the label for y_axis
    KNNType: 0 for equally weighted, 1 for weighted "linearly", 2 for modified KNN
Output
    graph of value against date
'''
def drawGraph(trueVal, predVal, yAxisLabel, weightedKNN):
    fig, ax = plt.subplots()
    x_axis = trueVal.index.tolist()
    ax.plot(x_axis, trueVal, label = "True", linewidth = 0.3, color = 'c')
    ax.plot(x_axis, predVal, label = "Predicted", linewidth = 0.3, color = 'm')
    string = "with weights" if weightedKNN == 1 else "without weights" if weightedKNN == 0 else "Modified KNN"
    plt.xlabel('Dates'), plt.ylabel(yAxisLabel), plt.legend(), plt.title('KNN_' + string)
    years = mdates.YearLocator()  # every month
    xFmt = mdates.DateFormatter('20%y')
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(xFmt)
    plt.savefig('KNN_' + yAxisLabel + '.png')
    plt.show()

'''
pData: processed data
Input
    d: dataset
Ouput
    returns arr of tuples, where first element is feature vector and the second element is the actual target
'''
def pData(d):
    arr = []
    for idx, row in d.iterrows():
        arr.append( ([itm for colName, itm in row.iteritems() if colName != 'Target'],  row['Target']) )
    return arr

'''
ED: euclidean distance
Input
    testData: test data point 
    processed: processed arrary / list
    k: parameter
    forecastType: from the parent function
    weights: by default is None
Ouput
    returns the predicted price
'''
def ED(testData, processed, k, forecastType, weights = None):
    distTrack, classTrack = [], []
    for idx, itr in enumerate(processed):
        arr, val = itr[0], itr[1]
        distTrack.append(sum([(a - b) ** 2.0 for a, b in zip(testData, arr)]))
    idx = np.argpartition(np.array(distTrack), k)
    for itr in idx[ : k]:
        classTrack.append(processed[itr][1])
    if weights == None:
        return sum(classTrack) / k if forecastType == 2 else mode(classTrack)
    else:
        avg = sum([a * b for a, b in zip(classTrack, weights)])
        return avg if forecastType == 2 else 1 if avg > 0 else - 1

'''
MED: modified euclidean distance (for MKNN)
Input
    testData: test data point 
    processed: processed array/ list
    k: parameter
    validity: validity of each training data point
Ouput
    returns the predicted price
'''
def MED(testData, processed, k, validity):
    distTrack, classTrack, validityTrack, weights = [], [], [], []
    for idx, itr in enumerate(processed):
        arr, val = itr[0], itr[1]
        distTrack.append(sum([(a - b) ** 2.0 for a, b in zip(testData, arr)]))
    track = np.argpartition(np.array(distTrack), k)
    for itr in track[ : k]:
        classTrack.append(processed[itr][1])
        validityTrack.append(validity[itr])
    #print(classTrack, validityTrack)
    for idx, itr in enumerate(validityTrack):
        weights.append(itr * 1 / (distTrack[idx] + 0.5))
    avg = sum([a * b for a, b in zip(classTrack, weights)]) 
    return 1 if avg > 0 else - 1