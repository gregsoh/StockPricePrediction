'''
This is the main code base. 
It is dependent on 
    |-EMA: which runs the moving averages algorithms (Method 1)
    |-KNN: which runs the K nearest neighbor algorithm (Method 2)
    |-DT: which runs the random forest algorithm (Method 3)
    |-FEATURE: which runs the random forest algorithm (Method 3)
    |-LSTM: which runs the LSTM algorithm (Method 4)
    This files are in turn dependent on
        |-DEBUG: which does some sanity check on data processing
        |-MEASURE: measuring tools including the Mean Square Error
'''
from MA import EMA, SMA #Method 1
from KNN import KNN #Method 2
from DT import DT #Method 3
from FEATURE import FEATURE #Method 3
from LSTM import LSTM #Method 4

# Other dependencies
from DEBUG import MAIN_CHECKDATA, MAIN_DOWNLOAD
from MEASURE import DATAINFO
from datetime import datetime
import fix_yahoo_finance as yf, matplotlib.pyplot as plt, matplotlib.dates as mdates, pandas as pd, random

'''
optimalWin: Determine best window size (and understanding how window size affects error)
Input
    dset: array of dataset we want to run
Output
    Graph of Log Mean Square Error against Window Size (we want log to observe better results)
'''
def optimalWin(dset):
    wSet = [5, 12, 26, 50, 100, 200] # Common window values: 5, 12, 26 are considered short period, 50 and 100 are considered mid period, 200 is considered long
    track = [[] for itr in range(len(dset) * 3)]
    for idx, d in enumerate(dset):  
        for w in wSet:
            a = 2 / (w + 1)
            track[idx * 3].append(SMA(d.Close, w, False, 1))
            track[idx * 3 + 1].append(EMA(d.Close, w, a, False, False, 1))
            track[idx * 3 + 2].append(EMA(d.Close, w, a, True, False, 1))
    mark, color = ['o', 'o', 'o', '^', '^', '^', '*', '*', '*', 'X', 'X', 'X'], ['c', 'm', 'y']

    for idx, itr in enumerate(track):
        plt.scatter(wSet, itr, marker = mark[idx], c = color[idx % 3], s = 15)
    plt.xlabel('Window Size'), plt.ylabel('Log Mean Square Error')
    plt.title('Effect of window size on accuracy', y = 1.08)
    plt.legend(['SP_SMA', 'SP_EMA', 'SP_DMA', 'APL_SMA', 'APL_EMA', 'APL_DMA', 'MID_SMA', 'MID_EMA', 'MID_DMA', 'GSK_SMA', 'GSK_EMA', 'GSK_DMA'], fontsize = 'x-small', loc = 'upper center', bbox_to_anchor = (0.5, 1.05), ncol = 4)
    plt.savefig('GRAPH_WSEFFECT.png')
    plt.show()

'''
movingAvg: Calling three moving averages (simple, exponential, double exponential)
Input
    w: size of window
    dset: array of dataset we want to run
    label: labels in the dataset
    a: alpha value (or smoothing factor)
    plot: do we want to plot a graph?
Output
    Graph of predicted prices against true prices (if we want to plot)
'''
def movingAvg(w, dset, label, a, plot):
    for idx, d in enumerate(dset):
        x_axis = dset[idx].index.tolist()
        y_true = d.Close
        y_SMA, SMAErr = SMA(d.Close, w, True), SMA(d.Close, w, False)
        y_EMA, EMAErr = EMA(d.Close, w, a, False, True), EMA(d.Close, w, a, False, False)
        y_DEMA, DEMAErr = EMA(d.Close, w, a, True, True), EMA(d.Close, w, a, True, False)
        print(label[idx] + " with a = " + str(a), "SMA Err: ", SMAErr, "; EMAErr: ", EMAErr, "; DEMAErr: ", DEMAErr)
        if plot:
            fig, ax = plt.subplots()
            ax.plot(x_axis, y_true, label = "True " + label[idx], linewidth = 0.3)
            ax.plot(x_axis, y_SMA, label = "SMA " + label[idx], linewidth = 0.3)
            ax.plot(x_axis, y_EMA, label = "EMA " + label[idx], linewidth = 0.3)
            ax.plot(x_axis, y_DEMA, label = "DMA " + label[idx], linewidth = 0.3)
            plt.xlabel('Dates'), plt.ylabel('Prices'), plt.legend(), plt.title('Moving Averages of ' + label[idx])
            months, xFmt = mdates.MonthLocator(), mdates.DateFormatter('%m, 20%y')
            ax.xaxis.set_major_locator(months), ax.xaxis.set_major_formatter(xFmt)
            plt.savefig('GRAPH_MAvg' + label[idx] + '.png')
            plt.show()
            plt.close()

'''
MAApp: Exponential Moving Average Application 
Input
    w_short and w_long: two different sizes of window
    d: data input
Output
    Graph of predicted prices against dates
Notes: to interpret, visit https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:moving_averages
'''
def MAApp(w_short, w_long, d):
    x_axis = d.index.tolist()
    y_short = EMA(d.Close, w_short, 2 / (w_short + 1), False, True)
    y_long = EMA(d.Close, w_long, 2 / (w_long + 1), False, True)
    plt.plot(x_axis, y_short, label = "Small window period", linewidth = 0.3, color = 'c')
    plt.plot(x_axis, y_long, label = "Large window period", linewidth = 0.3, color = 'g')
    plt.xlabel('Dates'), plt.ylabel('Prices'), plt.legend(), plt.title(str(w_short) + '-day and ' + str(w_long) + '-day moving averages')
    plt.xticks(rotation = 45), plt.tight_layout()
    plt.savefig('GRAPH_SHORTLONG.png')
    plt.show()

'''
KNearestNeighbor: Implementing KNN
Input 
    d: data
    k: parameter for the num of nearest neighbor we will be looking at (should be odd)
    numDays: predicting price numDays later
    forecastType: forecast boolean decision on whether to invest decision based on prices numDays days later (set to 1); forecast exact prices after numDays days (set to 2)
    plot: do we want to plot a graph?
Ouput
    return errors 
'''
def KNearestNeighbor(d, k, numDays, forecastType, plot):
    def target(d, numDays, type):
        tar = []
        for itr in range(len(d.Close) - numDays):
            if type == 1:
                tar.append(1) if d['Close'][itr + numDays] - d['Close'][itr] > 0 else tar.append(-1)
            else:
                tar.append(round(d['Close'][itr + numDays]))
        d = d[ : len(d) - numDays]
        d = d.assign(Target = tar)
        return d
    def normalize(d):
        d_Copy = d.copy()
        pop = {} # to keep track 
        for feature in d.columns:
            if feature != 'Target':
                maximum, minimum = d[feature].max(), d[feature].min()
                d_Copy[feature] = (d[feature] - minimum) / (maximum - minimum)
                pop[feature] = (minimum, maximum)
        return d_Copy, pop
      
    data = target(d, numDays, forecastType)
    data = data.round(2)
    train, test = data[ : (len(data) * 7) // 10], data[(len(data) * 7) // 10 : ] #70% train, 30% test
    train, pop = normalize(train)
    fig, ax = plt.subplots()
    if forecastType == 1:
        err1, predicted1 = KNN(train, test, pop, k, forecastType, False, False) #Normal
        err2, predicted2 = KNN(train, test, pop, k, forecastType, True, False) #Weighted
        err3, predicted3 = KNN(train, test, pop, k, forecastType, False, True) #Modified
        print("Forecast type " + str(forecastType) + " (normal, weighted, modified)", err1, err2, err3)
        if plot:
            x_axis = test.index.tolist()
            ax.plot(x_axis, test['Target'], label = "True", linewidth = 0.3, color = 'c')
            ax.plot(x_axis, predicted1, label = "Predicted - normal", linewidth = 0.3, color = 'm')
            ax.plot(x_axis, predicted2, label = "Predicted - weighted", linewidth = 0.3, color = 'g')
            ax.plot(x_axis, predicted3, label = "Predicted - modified", linewidth = 0.3, color = 'y')
            plt.xlabel('Dates'), plt.ylabel('Price will rise after ' + str(numDays) + ' days'), plt.legend(), plt.title('Different KNN on predicting whether prices will rise or fall')
            months, xFmt = mdates.MonthLocator(), mdates.DateFormatter('%m, 20%y')
            ax.xaxis.set_major_locator(months), ax.xaxis.set_major_formatter(xFmt), plt.xticks(rotation = 45), plt.tight_layout()
            plt.savefig('GRAPH_KNN1.png')
            plt.show()
        return err1, err2, err3
    else:
        err1, predicted1 = KNN(train, test, pop, k, forecastType, False, False) #Normal
        err2, predicted2 = KNN(train, test, pop, k, forecastType, True, False) #Weighted
        print("Forecast type " + str(forecastType) + " (normal, weighted)", err1, err2)
        if plot:
            x_axis = test.index.tolist()
            ax.plot(x_axis, test['Target'], label = "True", linewidth = 0.3, color = 'c')
            ax.plot(x_axis, predicted1, label = "Predicted - normal", linewidth = 0.3, color = 'm')
            ax.plot(x_axis, predicted2, label = "Predicted - weighted", linewidth = 0.3, color = 'y')
            plt.xlabel('Dates'), plt.ylabel('Price after ' + str(numDays) + ' days'), plt.legend(), plt.title('Different KNN on predicting actual prices')
            months, xFmt = mdates.MonthLocator(), mdates.DateFormatter('%m, 20%y')
            ax.xaxis.set_major_locator(months), ax.xaxis.set_major_formatter(xFmt), plt.xticks(rotation = 45), plt.tight_layout()
            plt.savefig('GRAPH_KNN2.png')
            plt.show()
        return err1, err2

'''
RF: random forests
Input 
    dset: list of data we are working with (non-truncated)
    forecastType: forecast boolean decision on whether to invest decision based on prices k days later (set to 1); forecast exact prices after k days (set to 2)
    label: labelling of the datas in the dataset
Ouput
    No return value
'''
def RF(dset, forecastType, label): 
    def smooth(d, a): #a is the smoothing factor
        numData = len(d)
        SD = [0] * numData
        for idx, itr in enumerate(d):
            SD[idx] = a * itr + (1 - a) * SD[idx - 1] if idx != 0 else d[0]
        return SD
    def target(d, numDays, type):
        tar = []
        for itr in range(len(d.Close) - numDays):
            if type == 1:
                tar.append(1) if d['Close'][itr + numDays] - d['Close'][itr] > 0 else tar.append(-1)
            else:
                tar.append(round(d['Close'][itr + numDays]))
        d = d[ : len(d) - numDays]
        d = d.assign(Target = tar)
        return d
    def bagging(d, numTrees, numTrain):
        numData = len(d)
        dataSet = []
        d = d.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis = 1)
        for itr in range(numTrees):
            dataSet.append(d.sample(n = numTrain, replace = True))
        return dataSet
    fig, ax = plt.subplots()
    for idx, data in enumerate(dset):
        k, errList = [itr for itr in range(5, 15)], []
        for days in k:
            #Step 1: split the data
            train, test = data[ : (len(data) * 7) // 10], data[(len(data) * 7) // 10 : ] #70% train, 30% test
            #Step 1: Smooth the data exponentially (note that we can and should only smooth the train data)
            train.assign(Close = smooth(train.Close, 0.2))
            #Step 2: Set the target
            train = target(train, days, forecastType)
            test = target(test, days, forecastType)
            #Step 3: Calculate the features
            features = ['RSI', 'SO', 'WIL', 'MACD', 'PRC', 'OBV']
            train, test = FEATURE(train), FEATURE(test)
            MAIN_CHECKDATA(train), MAIN_CHECKDATA(test)
            #Step 4: bagging/ bootstrap aggregating
            DS = bagging(train, 15, len(train) // 3) # Second parameter should be odd, otherwise we may receive some error
            #Step 5: Building the decision trees
            err, prediction = DT(DS, test, features, forecastType)
            errList.append(err)
        ax.plot(k, errList, label = label[idx], linewidth = 0.3)
    plt.xlabel('Different look-ahead periods'), plt.ylabel('Error'), plt.legend()
    plt.title('Random Forest, predicting whether prices will increase or fall') if forecastType == 1 else plt.title('Random Forest, predicting prices after k days')
    plt.xticks(rotation = 45), plt.tight_layout()
    plt.savefig('GRAPH_RF_MODIFYK1.png') if forecastType == 1 else plt.savefig('GRAPH_RF_MODIFYK2.png')
    plt.show()

'''
LONGSHORTTERMMEMEORY: Modified LSTM
Input
    dset: dataet
    label: tciker symbols of data
Output
    No return value
'''
def LONGSHORTTERMMEMEORY(dset, label):
    def smooth(d, a):
        numData = len(d)
        SD = [0] * numData
        for idx, itr in enumerate(d):
            SD[idx] = a * itr + (1 - a) * SD[idx - 1] if idx != 0 else d[0]
        return SD
    fig, ax = plt.subplots()   
    LOOKAHEAD = 3
    for idx, data in enumerate(dset):
        train, test = data[ : (len(data) * 7) // 10], data[(len(data) * 7) // 10 : ] #70% train, 30% test
        train = smooth(train, 0.2)
        err, true, pred = LSTM(train, test, LOOKAHEAD)
        x_axis = test.index.tolist()[ : len(true)] 
        ax.plot(x_axis, true, label = "True_" + label[idx], linewidth = 0.3)
        ax.plot(x_axis, pred, label = "Pred_" + label[idx], linewidth = 0.3)
        check = round(test[ : len(true)] / 10)
        print("The error for " + label[idx], err)
    plt.xlabel('Dates'), plt.ylabel('Prices'), plt.legend(), plt.title('LSTM for different stocks')
    months, xFmt = mdates.MonthLocator(), mdates.DateFormatter('%m, 20%y')
    ax.xaxis.set_major_locator(months), ax.xaxis.set_major_formatter(xFmt)
    plt.savefig('GRAPH_LSTM.png')
    plt.show()

'''
MainFn: main function
Input
    reload = Do we want to reload the data?
No Output
Note that we need to reload the data everytime we rerun the script
'''
def mainFn(reload):
    # TP stands for time period
    TP1 = ['2003-01-03', '2003-07-03'] # 6 months
    TP2 = ['2003-01-03', '2005-01-03'] # 2 years
    TP3 = ['2003-01-03', '2018-01-03'] # 15 years
    TP4 = ['1993-01-03', '2018-01-03'] # 25 years
    ticker = ['^GSPC', 'AAPL', 'MIDD', 'GSK']
    if reload: 
        #S: short TP; M: medium TP; L: long TP; E: extra long TP; GSPC: S&P; APL: Apple; MID: Middleby Corporation ;GSK: GlaxoSmithKlein
        SP_S, APL_S, MID_S, GSK_S = yf.download('^GSPC', TP1[0], TP1[1]), yf.download('AAPL', TP1[0], TP1[1]), yf.download('MIDD', TP1[0], TP1[1]), yf.download('GSK', TP1[0], TP1[1])
        SP_M, APL_M, MID_M, GSK_M = yf.download('^GSPC', TP2[0], TP2[1]), yf.download('AAPL', TP2[0], TP2[1]), yf.download('MIDD', TP2[0], TP2[1]), yf.download('GSK', TP2[0], TP2[1])
        SP_L, APL_L, MID_L, GSK_L = yf.download('^GSPC', TP3[0], TP3[1]), yf.download('AAPL', TP3[0], TP3[1]), yf.download('MIDD', TP3[0], TP3[1]), yf.download('GSK', TP3[0], TP3[1])
        SP_E, APL_E, MID_E, GSK_E = yf.download('^GSPC', TP4[0], TP4[1]), yf.download('AAPL', TP4[0], TP4[1]), yf.download('MIDD', TP4[0], TP4[1]), yf.download('GSK', TP4[0], TP4[1])
        ALLDATA = [SP_S, SP_M, SP_L, SP_E, APL_S, APL_M, APL_L, APL_E, MID_S, MID_M, MID_L, MID_E, GSK_S, GSK_M, GSK_L, GSK_E]
        MAIN_DOWNLOAD(ALLDATA)
        SP_S.to_pickle("./SPS.pkl"), APL_S.to_pickle("./APLS.pkl"), MID_S.to_pickle("./MIDS.pkl"), GSK_S.to_pickle("./GSKS.pkl")
        SP_M.to_pickle("./SPM.pkl"), APL_M.to_pickle("./APLS.pkl"), MID_M.to_pickle("./MIDM.pkl"), GSK_M.to_pickle("./GSKM.pkl")
        SP_L.to_pickle("./SPL.pkl"), APL_L.to_pickle("./APLL.pkl"), MID_L.to_pickle("./MIDL.pkl"), GSK_L.to_pickle("./GSKL.pkl")
        SP_E.to_pickle("./SPE.pkl"), APL_E.to_pickle("./APLE.pkl"), MID_E.to_pickle("./MIDE.pkl"), GSK_E.to_pickle("./GSKE.pkl")
    else:
        SP_S, APL_S, MID_S, GSK_S = pd.read_pickle("./SPS.pkl"), pd.read_pickle("./APLS.pkl"), pd.read_pickle("./MIDS.pkl"), pd.read_pickle("./GSKS.pkl")
        SP_M, APL_M, MID_M, GSK_M = pd.read_pickle("./SPM.pkl"), pd.read_pickle("./APLS.pkl"), pd.read_pickle("./MIDM.pkl"), pd.read_pickle("./GSKM.pkl")
        SP_L, APL_L, MID_L, GSK_L = pd.read_pickle("./SPL.pkl"), pd.read_pickle("./APLL.pkl"), pd.read_pickle("./MIDL.pkl"), pd.read_pickle("./GSKL.pkl")
        SP_E, APL_E, MID_E, GSK_E = pd.read_pickle("./SPE.pkl"), pd.read_pickle("./APLE.pkl"), pd.read_pickle("./MIDE.pkl"), pd.read_pickle("./GSKE.pkl")
        ALLDATA = [SP_S, SP_M, SP_L, SP_E, APL_S, APL_M, APL_L, APL_E, MID_S, MID_M, MID_L, MID_E, GSK_S, GSK_M, GSK_L, GSK_E]
        MAIN_DOWNLOAD(ALLDATA)

    #Part 1: How window size affects MSE
    def partOne():
        data = [SP_L, APL_L, MID_L, GSK_L]
        optimalWin(data)
        for itr in data:
            m, s = DATAINFO(itr.Close)
            print("Mean and standard deviation: ", m, s)

    #Part 2: Moving averages -- how different moving averages perform
    def partTwo():
        w = 26 #known  to be common
        a1, a2, a3, a4 = 2 / (w + 1), 4 / (w + 1), 8 / (w + 1), 16 / (w + 1)
        data = [SP_S, APL_S, MID_S, GSK_S]
        movingAvg(w, data, ticker, a1, True)
        movingAvg(w, data, ticker, a2, False), movingAvg(w, data, ticker, a3, False), movingAvg(w, data, ticker, a4, False) #Trying to see how smoothing factor affects (err values will be printed out)

    #Part 3: Moving average application
    def partThree():
        w_short, w_long = 12, 100
        MAApp(w_short, w_long, SP_M)

    #Part 4: K Nearest Neighbor
    def partFour():
        typeOneK, typeTwoK, typeOneDays, typeTwoDays = [], [], [], [] #keep track of normal KNN
        # Modify K
        x_axis_K = [itr for itr in range(3, 17, 2)]
        for itr in x_axis_K:
            if itr == 3: 
                err1, err2, err3 = KNearestNeighbor(SP_M, itr, 5, 1, True)
                err4, err5 = KNearestNeighbor(SP_M, itr, 5, 2, True)
                typeOneK.append(err1), typeTwoK.append(err4)
            else:
                err1, err2, err3 = KNearestNeighbor(SP_M, itr, 5, 1, False)
                err4, err5 = KNearestNeighbor(SP_M, itr, 5, 2, False)
                typeOneK.append(err1), typeTwoK.append(err4)
        plt.plot(x_axis_K, typeOneK, label = "Predicting whether prices fall or rise", linewidth = 0.3, color = 'c')
        plt.plot(x_axis_K, typeTwoK, label = "Predicting prices", linewidth = 0.3, color = 'm')
        plt.xlabel('Value of K'), plt.ylabel('Error'), plt.legend(), plt.title('Error for different values of K')
        plt.savefig('GRAPH_KNN_MODIFYK.png')
        plt.close()
        # Modify numDays
        x_axis_Days = [itr for itr in range(5, 55, 5)]
        for itr in x_axis_Days:
            err1, err2, err3 = KNearestNeighbor(SP_M, 5, itr, 1, False)
            err4, err5 = KNearestNeighbor(SP_M, 5, itr, 2, False)
            typeOneDays.append(err1), typeTwoDays.append(err4)
        plt.plot(x_axis_Days, typeOneDays, label = "Predicting whether prices fall or rise", linewidth = 0.3, color = 'c')
        plt.plot(x_axis_Days, typeTwoDays, label = "Predicting prices", linewidth = 0.3, color = 'm')
        plt.xlabel('Predicting number of days in advance'), plt.ylabel('Error'), plt.legend(), plt.title('Error for different look-ahead periods')
        plt.savefig('GRAPH_KNN_MODIFYDAYS.png')
        plt.close()

    #Part 5: Random Forest
    def partFive():
        data = [SP_M, APL_M, MID_M, GSK_M]
        RF(data, 1, ticker)
        RF(data, 2, ticker)

    #Part 6: LSTM
    def partSix():
        data = [SP_M.Close]
        LONGSHORTTERMMEMEORY(data, ticker)
    partSix()
mainFn(False)