'''
This file contains the implementation of LSTM (The main function is LSTM)
Reference:https://hackernoon.com/understanding-architecture-of-lstm-cell-from-scratch-with-code-8da40f0b71f4
Reference: http://wp.firrm.de/index.php/2018/04/13/building-a-lstm-network-completely-from-scratch-no-libraries/
'''
import numpy as np, math
from MEASURE import MSE
from DEBUG import LSTM_CHECK, LSTM_CHECKDIM, GENERAL_CHECKSIZE, GENERAL_CHECKEMPTY, GENERAL_CHECKVAL, LSTM_NAN

'''
Paramters
'''
H = 64 #size of hidden state; LSTM has only one layer of hidden neuron
B = 10 #number of batches
D = None #dimensionality of data (divide price by 10 to get index into the dimensions)
SCALELINEAR = None
BUFFERCONST = 200 #If test set out of range, increase this value
LAMBDA = 0.01 #Learning rate
NUMITER = 2000

'''
LSTM: Main function
input
    train: train dataset
    test: test dataset
    numDays: predicting number of days ahead
return 
    MSE of test set, true value and predicted value
'''
def LSTM(train, test, numDays):
    def create_dataset(dataset, nD, train):
        SCALECONST = 1
        dataX, dataY = [], []
        if train:
            global D
            global SCALELINEAR
            SCALELINEAR = int(round(min(dataset)))
            D = int(round(max(dataset)) - SCALELINEAR + BUFFERCONST)
        for itm in range(len(dataset) - nD):
            dataX.append(int(round(dataset[itm] / SCALECONST) - SCALELINEAR + BUFFERCONST / 2) )
            dataY.append(int(round(dataset[itm + nD] / SCALECONST) - SCALELINEAR + BUFFERCONST / 2) )
        return np.array(dataX), np.array(dataY)
    def MSEMODIFY(t, p):
        err = 0
        for idx, itr in enumerate(t):
            err += MSE(itr, p[idx])
        return err
    def batchToList(l):
        numElem = len(l)
        rval = []
        for b in range(B):
            for x in range(numElem):
                rval.append(l[x][b])
        return rval

    LSTM_NAN(test), LSTM_NAN(train)
    trainX, trainY = create_dataset(train, numDays, True)
    grads, h, c = trainfn(trainX, trainY, NUMITER)
    testX, testY = create_dataset(test, numDays, False)
    pred, true = testfn(grads, h, c, testX, testY)
    return MSEMODIFY(true, pred), batchToList(true), batchToList(pred)

'''
Gradient: For RMS prop
Input
    No input
No return value
'''
def gradient():
    GRAD = {}
    GRAD['GWf'] = np.zeros((D + H, H))
    GRAD['GWi'] = np.zeros((D + H, H))
    GRAD['GWc'] = np.zeros((D + H, H))
    GRAD['GWo'] = np.zeros((D + H, H))
    GRAD['GWy'] = np.zeros((H, D))
    GRAD['Gbf'], GRAD['Gbi'], GRAD['Gbc'], GRAD['Gbo'], GRAD['Gby'] = np.zeros((B, H)), np.zeros((B, H)), np.zeros((B, H)), np.zeros((B, H)), np.zeros((B, D))
    return GRAD

'''
trainfn: the main function that calls forward and backward propagation
Input
    trainX: train data
    trainY: true predicted values
    numIter: number of iterations
Output
    returns accuracy (or MSE)
'''
def trainfn(trainX, trainY, numIter):
    def process(train):
        numElem = len(train) // B #should be len of array 
        START, TRUNCONST = 0, 10
        dataM = []
        for itr in range(B):
            series = train[START : START + numElem]
            LSTM_NAN(series)
            dataM.append(series)
            START += numElem
        nonOneHotArr, oneHotArr = [], []
        for itr in range(numElem):
            oneHot, nonOneHot = np.zeros((B, D)), []
            for b in range(B):
                number = dataM[b][itr]
                nonOneHot.append(number)
            for idx, o in enumerate(nonOneHot):
                oneHot[idx, o] = 1.
            nonOneHotArr.append(nonOneHot)
            oneHotArr.append(oneHot)
        return nonOneHotArr, oneHotArr
    def update(grad, params, GRAD):
        # Utilizing RMS prop
        GWf, GWi, GWc,GWo, GWy, Gbf, Gbi, Gbc, Gbo, Gby =  GRAD['GWf'], GRAD['GWi'], GRAD['GWc'], GRAD['GWo'], GRAD['GWy'], GRAD['Gbf'], GRAD['Gbi'], GRAD['Gbc'], GRAD['Gbo'], GRAD['Gby']
        beta = 0.9
        GWf = GWf * beta + (grad["Wf"] ** 2) * (1 - beta)
        GWi = GWi * beta + (grad["Wi"] ** 2) * (1 - beta)
        GWc = GWc * beta + (grad["Wc"] ** 2) * (1 - beta)
        GWo = GWo * beta + (grad["Wo"] ** 2) * (1 - beta)
        GWy = GWy * beta + (grad["Wy"] ** 2) * (1 - beta)
        Gbf = Gbf * beta + (grad["bf"] ** 2) * (1 - beta)
        Gbi = Gbi * beta + (grad["bi"] ** 2) * (1 - beta)
        Gbc = Gbc * beta + (grad["bc"] ** 2) * (1 - beta)
        Gbo = Gbo * beta + (grad["bo"] ** 2) * (1 - beta)
        Gby = Gby * beta + (grad["by"] ** 2) * (1 - beta)
        GRAD['GWf'], GRAD['GWi'], GRAD['GWc'], GRAD['GWo'], GRAD['GWy'], GRAD['Gbf'], GRAD['Gbi'], GRAD['Gbc'], GRAD['Gbo'], GRAD['Gby'] = GWf, GWi, GWc,GWo, GWy, Gbf, Gbi, Gbc, Gbo, Gby
        params["Wf"] -= LAMBDA / np.sqrt(GWf + 1e-08) * grad["Wf"]
        params["Wi"] -= LAMBDA / np.sqrt(GWi + 1e-08) * grad["Wi"]
        params["Wc"] -= LAMBDA / np.sqrt(GWc + 1e-08) * grad["Wc"]
        params["Wo"] -= LAMBDA / np.sqrt(GWo + 1e-08) * grad["Wo"]
        params["Wy"] -= LAMBDA / np.sqrt(GWy + 1e-08) * grad["Wy"]
        params["bf"] -= LAMBDA / np.sqrt(Gbf + 1e-08) * grad["bf"]
        params["bi"] -= LAMBDA / np.sqrt(Gbi + 1e-08) * grad["bi"]
        params["bc"] -= LAMBDA / np.sqrt(Gbc + 1e-08) * grad["bc"]
        params["bo"] -= LAMBDA / np.sqrt(Gbo + 1e-08) * grad["bo"]
        params["by"] -= LAMBDA / np.sqrt(Gby + 1e-08) * grad["by"]
        return params, GRAD
    GENERAL_CHECKSIZE(trainX, trainY, "(LSTM.py: trainX and trainY do not have the same length)")
    nonOneHotX, oneHotX = process(trainX) 
    nonOneHotY, oneHotY = process(trainY)
    GENERAL_CHECKEMPTY(oneHotX, '(LSTM.py: No processed data)'), GENERAL_CHECKEMPTY(nonOneHotX, '(LSTM.py: No processed data)'), LSTM_CHECKDIM(oneHotX, len(trainX) // B, B, D)
    GENERAL_CHECKEMPTY(oneHotY, '(LSTM.py: No processed data)'), GENERAL_CHECKEMPTY(nonOneHotY, '(LSTM.py: No processed data)'), LSTM_CHECKDIM(oneHotY, len(trainY) // B, B, D)
    parameters = weightInitialization()
    GRAD = gradient()
    h, c = np.zeros((B, H)), np.zeros((B, H))
    for itr in range(numIter):
        grads, loss, h, c = trainStep(oneHotX, oneHotY, nonOneHotY, parameters, h, c, itr)
        parameters, GRAD = update(grads, parameters, GRAD)
        if itr % 50 == 0:
            print("Itertion", str(itr), loss)
    return parameters, h, c

'''
weightInitialization: initializing weights
Input
    No input
Output
    returns dictionary with random values
'''
def weightInitialization():
    mean, STD, pop = 0, 0.01, {}
    pop["Wf"] = np.random.normal(mean, STD, (D + H,  H))
    pop["Wi"] = np.random.normal(mean, STD, (D + H,  H))
    pop["Wc"] = np.random.normal(mean, STD, (D + H,  H))
    pop["Wo"] = np.random.normal(mean, STD, (D + H,  H))
    pop["Wy"] = np.random.normal(mean, STD, (H,  D))
    pop["bf"], pop["bi"], pop["bc"], pop["bo"], pop["by"]  = np.zeros((B, H)), np.zeros((B, H)), np.zeros((B, H)), np.zeros((B, H)), np.zeros((B, D))
    return pop

'''
activatingFn: activating functions used throughout
Input
    idx: index of activiating function to use
    val: value we are transforming (accepts 1, 2, 3, 4 or 5)
Output
    returns value
'''
def activatingFn(idx, val):
    def sig(val):
        return 1 / (1 + np.exp(-val))
    def tanh(val):
        return np.tanh(val)
        #Original function: return (np.exp(val) - np.exp(- val)) / (np.exp(val) + np.exp(- val))
    def sMax(val): 
        rval = np.exp(val)
        ttl = np.sum(rval, axis = 1).reshape(-1, 1)
        return  rval / ttl
    def tanhDerivative(val):
        return 1. - np.tanh(val) ** 2.
    def sigDerivative(val):
        return np.multiply(sig(val), 1 - sig(val))
    
    if idx == 1:
        return sig(val) #sigmoid
    elif idx == 2:
        return tanh(val) #tangent function
    elif idx == 3:
        return sMax(val) #softMax
    elif idx == 4:
        return tanhDerivative(val) #derivative of tanh
    else:
        return sigDerivative(val) #derivative of sigmoid

'''
trainStep: each training step
Input
    batches: takes in batches of data
    trueH: true data (hot)
    true: true data (non one-hot)
    params: takes in the dictionary of parameters we will be updating
    h: activation updated
    c: cell updated
Output
    caches and output
'''
def trainStep(batches, trueH, true, params, h, c, itr):
    def oneHotToNum(output): # converting one hot representation to actual output bto calculate err
        pred = []
        for itr in output:
            idx = list(itr).index(max(list(itr)))
            GENERAL_CHECKVAL(idx, 0, D, '(LSTM.py: something wrong with one-hot representation)')
            pred.append(idx)
        return pred
    def CrossEntropy(t, p):
        err = 0 
        for itr in p:
            idx = list(itr).index(max(list(itr)))
            GENERAL_CHECKVAL(idx, 0, D, '(LSTM.py: something wrong with index)')
            val = itr[idx]
            if val == 0.:
                print(val, itr)
                raise ValueError('LSTM.py: Val seems to be out of range')
            err += -math.log2(val) 
        return err
    caches, output = [], []
    loss = 0.
    for b, t in zip(batches, true):
        h, c, o, cache = forwardPropagation(b, h, c, params)
        loss += CrossEntropy(t, o)
        output.append(o)
        caches.append(cache)
    loss = loss / len(batches)
    h_next, c_next = np.zeros((B, H)), np.zeros((B, H))
    grads = {k : np.zeros_like(v) for k, v in params.items()} #This is a copy of weight initialization that we are training
    for o, t, ca in reversed(list(zip(output, true, caches))):
        grad, h_next, c_next = backwardPropagation(o, np.asarray(t), ca, h_next, c_next)
        for k in grads.keys(): #Updating gradient
            grads[k] += grad[k] / len(batches)
    return grads, loss, h, c

'''
forwardPropagation: LSTM cell and computation
Input
    data: one batch data in one-hot format
    h_Old: previous activation matrix
    c_Old: previous cell matrix
    param: dictionary of parameters
Output
    returns newCell, newActivation, newOutput to be used for next time step
'''
def forwardPropagation(data, h_Old, c_Old, param):
    Wf, Wi, Wc, Wo, Wy = param["Wf"], param["Wi"], param["Wc"], param["Wo"], param["Wy"]
    bf, bi, bc, bo, by = param["bf"], param["bi"], param["bc"], param["bo"], param["by"]
    #forget gate
    X = np.hstack((h_Old, data)) #dimension of precActivation is B x H, batchData is B x D
    hf = activatingFn(1, np.matmul(X, Wf) + bf) #dimension of weightFG = (H + D) x H
    #input gate
    hi = activatingFn(1, np.matmul(X, Wi) + bi) #layer 1
    hc = activatingFn(2, np.matmul(X, Wc) + bc) #layer 2
    #ouput gate
    ho =  activatingFn(1, np.matmul(X, Wo) + bo) #dimension is B x H
    c = np.multiply(hf, c_Old) + np.multiply(hi, hc) #dimenion of newCell B x H
    h = np.multiply(ho, activatingFn(2, c)) # dimension is B x H
    o = activatingFn(3, np.matmul(h, Wy) + by) #dimension of newOutput B x D
    cache = {}
    cache['X'], cache['hf'], cache['hi'], cache['hc'], cache['ho'], cache['c'], cache['h'], cache['o'], cache['h_Old'], cache['c_Old'] = X, hf, hi, hc, ho, c, h, o, h_Old, c_Old
    cache["Wf"], cache["Wi"], cache["Wc"], cache["Wo"], cache["Wy"] = Wf, Wi, Wc, Wo, Wy
    cache["bf"], cache["bi"], cache["bc"], cache["bo"], cache["by"] = bf, bi, bc, bo, by
    return h, c, o, cache

'''
backwardPropagation: the function that handles all the derivative
Input
    pred: predicted value
    true: true value
    cache: cache of all the variable sin weight initialization (so that we can keep track and update)
    dh_next: derivative of h_next
    dc_next: derivative of c_next
Output
    grads: the update that needs to make to gradients
    dh_next and dc_next: derivative of h and c for next time step
'''
def backwardPropagation(pred, true, cache, dh_next, dc_next):
    X, hf, hi, hc, ho, c, h, o, h_old, c_old = cache['X'], cache['hf'], cache['hi'], cache['hc'], cache['ho'], cache['c'], cache['h'], cache['o'], cache['h_Old'], cache['c_Old']
    Wf, Wi, Wc, Wo, Wy = cache["Wf"], cache["Wi"], cache["Wc"], cache["Wo"], cache["Wy"]
    bf, bi, bc, bo, by = cache["bf"], cache["bi"], cache["bc"], cache["bo"], cache["by"]
    # Softmax loss gradient
    dy = pred.copy()
    true = true.reshape((1, B))
    m = true.shape[0]
    dy[range(m), true] -= 1.
    # dy = dy / m
    # Hidden to output gradient
    dWy = np.matmul(h.T, dy)
    dby = dy
    dh = np.matmul(dy, Wy.T) + dh_next #dh_next has dimensions B x H
    # Gradient for ho in h = ho * tanh(c)
    dho = activatingFn(2, c) * dh
    dho = activatingFn(5, ho) * dho

    # Gradient for c in h = ho * tanh(c)
    dc = ho * dh 
    dc = dc * activatingFn(4, c)
    dc = dc + dc_next

    # Gradient for hf in c = hf * c_old + hi * hc
    dhf = c_old * dc
    dhf = activatingFn(5, hf) * dhf

    # Gradient for hi in c = hf * c_old + hi * hc
    dhi = hc * dc
    dhi = activatingFn(5, hi) * dhi

    # Gradient for hc in c = hf * c_old + hi * hc
    dhc = hi * dc
    dhc = activatingFn(4, hc) * dhc

    # Gate gradients, just a normal fully connected layer gradient
    dWf = np.matmul(X.T, dhf)
    dbf = dhf
    dXf = np.matmul(dhf, Wf.T)

    dWi = np.matmul(X.T, dhi)
    dbi = dhi
    dXi = np.matmul(dhi, Wi.T)

    dWo = np.matmul(X.T, dho)
    dbo = dho
    dXo = np.matmul(dho, Wo.T)

    dWc = np.matmul(X.T, dhc)
    dbc = dhc
    dXc = np.matmul(dhc, Wc.T)
    # Accumulating gradients here
    dX = dXo + dXc + dXi + dXf

    # Splitting the concatenated matrix
    dh_next = dX[:, :H]
    # Gradient for c_old in c = hf * c_old + hi * hc
    dc_next = hf * dc

    grad = dict(Wf = dWf, Wi = dWi, Wc = dWc, Wo = dWo, Wy = dWy, bf = dbf, bi = dbi, bc = dbc, bo = dbo, by = dby)
    return grad, dh_next, dc_next

'''
testfn: function that tests the unseen data
Input
    grads: trained grads
    h: the learned h
    c: the learned c
    testX: test inputs
    testY: true val
Output
    Predicted values, true values
'''
def testfn(grads, h, c, testX, testY):

    def process(d):
        numElem = len(d) // B #should be len of array 
        START, TRUNCONST = 0, 10
        dataM = []
        for itr in range(B):
            series = d[START : START + numElem]
            LSTM_NAN(series)
            dataM.append(series)
            START += numElem
        nonOneHotArr, oneHotArr = [], []
        for itr in range(numElem):
            oneHot, nonOneHot = np.zeros((B, D)), []
            for b in range(B):
                number = dataM[b][itr]
                nonOneHot.append(number)
            for idx, o in enumerate(nonOneHot):
                oneHot[idx, o] = 1.
            nonOneHotArr.append(nonOneHot)
            oneHotArr.append(oneHot)
        return nonOneHotArr, oneHotArr
    def oneHotToNum(output): # converting one hot representation to actual output to calculate err
        pred = []
        for itr in output:
            idx = list(itr).index(max(list(itr)))
            GENERAL_CHECKVAL(idx, 0, D, '(LSTM.py: something wrong with one-hot representation)')
            pred.append(idx - BUFFERCONST / 2 + SCALELINEAR)
        return pred
    pred = []
    nonOneHotX, oneHotX = process(testX)
    nonOneHotY, oneHotY = process(testY)
    for idx, itr in enumerate(nonOneHotY):
        for idx2, itr2 in enumerate(itr):
            nonOneHotY[idx][idx2] = itr2 - BUFFERCONST / 2 + SCALELINEAR
    for itr, itm in enumerate(oneHotX):
        _1 , _2 , o, _3 = forwardPropagation(itm, h, c, grads)
        pred.append(oneHotToNum(o))
    return pred, nonOneHotY