'''
This file handles most of the main code for the implementation of the random forest
'''
from MEASURE import MSE
from DEBUG import DT_CHECKVALUES
from statistics import mode, mean
import matplotlib.pyplot as plt, pandas as pd, random, matplotlib.dates as mdates, numpy as np, math

'''
RF: Random Forest main code
Input
    train: train set
    test: test set
    features: features we are considering
    forecastType: forecast type (1 if its boolean decision to invest based on prices k days later and 2 if its predicting the price k days later)
Output
    returns mean squared error and the predicted value
'''
def DT(train, test, features, forecastType):
    trueVal, target = test['Target'], 'Target'
    numTrees = len(train)
    track, predVal = [[] for itr in range(len(trueVal))], [0] * len(trueVal)
    for itr in range(numTrees):
        random.shuffle(features)
        if forecastType == 1: 
            builtTree = ID3(train[itr], train[itr], features[ : len(features) // 2], True)
        else:
            builtTree = ID3(train[itr], train[itr], features[ : len(features) // 2], False)
        try:
            default = mode(train[itr][target])
        except:
            default = mean(train[itr][target]) if forecastType == 2 else random.choice([-1, 1])
        pred = findingPred(test, builtTree, default)
        if len(pred) == 0:
            print(builtTree, test)
        DT_CHECKVALUES(pred, False)
        for idx, itm in enumerate(pred):
            track[idx].append(itm)
    for idx, itr in enumerate(track):
        try: 
            predVal[idx] = mode(itr)
        except:
            predVal[idx] = random.choice([-1, 1]) if forecastType == 1 else mean(itr)
    DT_CHECKVALUES(predVal, forecastType == 1)
    return MSE(trueVal, predVal), predVal

'''
findingPred: Finding predicted value on unseen data
Input
    testData: test dataset
    tree: builtTree from ID3
    default: if the tree does not contain the value, we will predict based on mode of target attribute
Output
    returns predicted value
'''
def findingPred(testData, tree, default):
    predVal = pd.DataFrame(columns = ['predicted'])
    queries = testData.iloc[: , : ].to_dict(orient = "records")
    for itr in range(len(testData)):
        predVal.loc[itr, "predicted"] = predict(queries[itr], tree, default) 
    return predVal['predicted']

'''
Predict: helper function for findingPred
Input
    queries: test dataset
    tree: builtTree from ID3
    default: if the tree does not contain the value, we will predict based on mode of target attribute
Output
    returns predicted value
'''
def predict(q, tree, default):
    for key in list(q.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][q[key]] 
            except:
                return default
            result = tree[key][q[key]]
            if isinstance(result, dict):
                return predict(q, result, default)
            else:
                if type(result) != int and type(result) != float and type(result) != np.int64 and type(result) != np.float64:
                    print(result, type(result))
                    raise ValueError('DT.py: result in predict function is not an integer!')
                return result

'''
ID3: using ID3 algorithm to build tree, uses Entropy and Information Gain
Input
    data: spliced data based on node conditions
    oriData: the original data set
    features: the features we will consider (decreases with each depth of tree)
    binary: is this binary classifier
    parentNode: to keep track of our parent 
Output
    tree represented in dictionary form
'''
def ID3(data, oriData, features, binary, parentNode = None):
    target = "Target"
    if len(set(data[target])) == 1: # only one possible value
        return data[target][0]
    elif len(data) == 0: # if there is no more data left
        return mode(oriData[target])
    elif len(features) == 0:
        return parentNode
    else:
        try:
            parentNode = mode(data[target])
        except:
            parentNode = mean(data[target]) if not binary else random.choice([-1, 1]) 
        eT = entropyTarget(data, target)
        r = [eT - entropyAttributes(data, f, target) for f in features]
        bestFeature = features[r.index(max(r))]
        tree = {bestFeature: {} }
        newFeatures = [feature for feature in features if feature != bestFeature]
        uniqueVal = set()
        for val in data[bestFeature]:
            uniqueVal.add(val)
        for val in uniqueVal:
            subData = data.loc[data[bestFeature] == val]
            subtree = ID3(subData, oriData, newFeatures, binary, parentNode)
            tree[bestFeature][val] = subtree
        return tree

'''
entropyTarget: Calculates entropy based on the target
Input
    train: train data (directly from RF)
    target: val we are predicting is "Target"
Output
    entropy value
'''
def entropyTarget(train, target):
    pop, ttl, rv, temp = {}, len(train[target]), 0, 0 #rv stands for returnValue
    for itr in train.index:
        val = None
        if type(train.at[itr, target]) == np.ndarray:
            mod = 0. if train.at[itr, target][0] == -0. else train.at[itr, target][0] 
            val = str(mod)
        else:
            mod = 0. if train.at[itr, target] == -0. else train.at[itr, target] 
            val = str(mod)
        temp = pop[val] if val in pop else 0
        pop[val] = temp + (1 / ttl)
    for key, val in pop.items():
        rv += - val * math.log2(val)
    return rv

'''
entropyAttribute: Calculates entropy on attributes
Input
    train: train data (directly from RF)
    numFac: number of attributes
    target: target column
Output
    entropy value
'''
def entropyAttributes(train, feature, target):
    def calculateEntropy(data, colName, target):
        rv, pop, ttl, temp = 0, {}, len(data.Close), 0
        for itr in data.index:
            val = None
            if type(data.at[itr, colName]) == np.ndarray:
                mod = 0. if data.at[itr, colName][0] == -0. else data.at[itr, colName][0] 
                val = str(mod)
            else:
                mod = 0. if data.at[itr, colName] == -0. else data.at[itr, colName] 
                val = str(mod)
            temp = pop[val] if val in pop else 0
            pop[val] = temp + (1 / ttl)
        for key, val in pop.items():
            rv += int(val) * entropyTarget(data.loc[data[colName] == float(key)], target)
        return rv
    return calculateEntropy(train, feature, target)