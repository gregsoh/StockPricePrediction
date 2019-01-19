import math
from MSE import MSE
from statistics import mode, mean
import pandas as pd
# RF: Random Forest main code
# Input
#   train: train set
#   test: test set
#   features: features we are considering
# Output
#   returns mean squared error
def DT(train, test, features):
    builtTree = ID3(train, train, features)
    try:
        default = mode(train['Close'])
    except:
        default = mean(train['Close'])
    return findingAccuracy(test, builtTree, default)

# findingAccuracy: using MSE to calculate the error on unseen data
# Input
#   testData: test dataset
#   tree: builtTree from ID3
#   default: if the tree does not contain the value, we will predict based on mode of target attribute
# Output
#   returns mean squared error
def findingAccuracy(testData, tree, default):
    trueVal, predVal = testData['Close'], pd.DataFrame(columns = ['predicted'])
    queries = testData.iloc[: , :-1].to_dict(orient = "records")
    for itr in range(len(testData)):
        predVal.loc[itr, "predicted"] = predict(queries[itr], tree, default) 
    return MSE(trueVal, predVal['predicted'])

# Predict: takes into account 
# Input
#   queries: test dataset
#   tree: builtTree from ID3
#   default: if the tree does not contain the value, we will predict based on mode of target attribute
# Output
#   returns predicted value
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
                return result

# ID3: using ID3 algorithm to build tree which uses Entropy and Information Gain
# Input
#   data: spliced data based on node conditions
#   oriData: the original data set
#   features: the features we will consider (decreases with each depth of tree)
#   target: val we are predicting is "Closing Price"
#   parentNode: to keep track of our parent 
# Output
#   tree represented in dictionary form
def ID3(data, oriData, features, target = "Close", parentNode = None):
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
            parentNode = mean(data[target])
        eT = entropyTarget(data)
        r = [eT - entropyAttributes(data, f) for f in features]
        bestFeature = features[r.index(max(r))]
        tree = {bestFeature: {} }
        features = [feature for feature in features if feature != bestFeature]
        uniqueVal = set()
        for val in data[bestFeature]:
            uniqueVal.add(val)
        for val in uniqueVal:
            subData = data.loc[data[bestFeature] == val]
            subtree = ID3(subData, oriData, features, target, parentNode)
            tree[bestFeature][val] = subtree
        return tree

# entropyTarget: Calculates entropy based on the target
#   target (stock prices) discretized to nearest integers
# Input
#   train: train data (directly from RF)
# Output
#   entropy value
def entropyTarget(train):
    pop, ttl, rv = {}, len(train.Close), 0 #rv stands for returnValue
    for itr in train.Close:
        pop[itr] = pop.get(itr, 0) + (1 / ttl)
    for key, val in pop.items():
        rv += - val * math.log2(val)
    return rv

# entropyAttribute: Calculates entropy on attributes
# Input
#   train: train data (directly from RF)
#   numFac: number of attributes
# Output
#   entropy value
def entropyAttributes(train, feature):
    def calculateEntropy(data, colName):
        rv, pop, ttl = 0, {}, len(train.Close)
        for itr in data[colName]:
            pop[itr] = pop.get(itr, 0) + (1 / ttl)
        for key, val in pop.items():
            rv += val * entropyTarget(train.loc[train[colName] == key])
        return rv
    return calculateEntropy(train, feature)