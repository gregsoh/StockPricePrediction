from MA import EMA, SMA
from FEATURE import FEATURE
from DT import DT
from DEBUG import CHECKDATA
from datetime import datetime
import fix_yahoo_finance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import random
'''
optimalWinSize: Trying to determine the best window size (and understanding how window size affects)
Input
    data: data we give
Output
    Graph of Mean Square Error against Window Size
'''
def optimalWinSize(data):
    winSize = [5, 12, 26, 50, 100 , 200] #These are common window values
    SMAErr, EMAErr, DEMAErr = [], [], []
    for itr in winSize:
        SMAErr.append(SMA(data.Close, itr, False))
        EMAErr.append(EMA(data.Close, itr, False, False))
        DEMAErr.append(EMA(data.Close, itr, True, False))
    plt.scatter(winSize, SMAErr), plt.scatter(winSize, EMAErr), plt.scatter(winSize, DEMAErr)  
    plt.xlabel('Window Size'), plt.ylabel('Mean Square Error')
    plt.title('Moving Average')
    plt.legend(['S&P SMA', 'S&P EMA', 'S&P DEMA'], loc='upper right')
    plt.savefig('OWS.png')
    plt.show()

'''
movingAverages: Calling three moving average (simple, exponential, double exponential)
Input
    winSize: size of window
    data: data we give
Output
    Graph of predicted prices against true prices
'''
def movingAverages(winSize, data):
    x_axis = data.index.tolist()[winSize * 2 : ]
    for idx, itr in enumerate(x_axis):
        x_axis[idx] = itr.date()
    y_true = data.Close[winSize * 2 : ]
    y_SMA, SMAErr = SMA(data.Close, winSize, True)[winSize:], SMA(data.Close, winSize, False)
    y_EMA, EMAErr = EMA(data.Close, winSize, False, True)[winSize:], EMA(data.Close, winSize, False, False)
    y_DEMA, DEMAErr = EMA(data.Close, winSize, True, True), EMA(data.Close, winSize, True, False)
    fig, ax = plt.subplots()
    ax.plot(x_axis, y_true, label = "True", linewidth = 0.3, color = 'c')
    ax.plot(x_axis, y_SMA, label = "SMA", linewidth = 0.3, color = 'm')
    ax.plot(x_axis, y_EMA, label = "EMA", linewidth = 0.3, color = 'y')
    ax.plot(x_axis, y_DEMA, label = "DEMA", linewidth = 0.3, color='g')
    plt.xlabel('Dates'), plt.ylabel('Prices'), plt.legend(), plt.title('Moving Averages')
    months = mdates.MonthLocator()  # every month
    xFmt = mdates.DateFormatter('%m, 20%y')
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(xFmt)
    plt.savefig('MA.png')
    print("SMA Err: ", SMAErr, "; EMAErr: ", EMAErr, "; DEMAErr: ", DEMAErr)
    plt.show()

'''
# MAApplication: Exponential Moving average Application
Input
    winSizeShort and winSizeLong: two different sizes of window
    data: data we give
Output
    Graph of predicted prices against dates
Notes: to interpret what this chart tells us, visit https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:moving_averages
'''
def MAApplication(winSizeShort, winSizeLong, data):
    x_axis = data.index.tolist()[winSizeLong : ]
    y_short = EMA(data.Close, winSizeShort, False, True)[winSizeLong - winSizeShort : ]
    y_long = EMA(data.Close, winSizeLong, False, True)
    plt.plot(x_axis, y_short, label = "Small window period", linewidth = 0.3, color = 'c')
    plt.plot(x_axis, y_long, label = "Large window period", linewidth = 0.3, color = 'g')
    plt.xlabel('Dates'), plt.ylabel('Prices'), plt.legend(), plt.title('One short, one long moving average')
    plt.savefig('ShortLong.png')
    plt.show()

'''
RF: random forests
Input 
    data: data we want to work with 
    forecastType: forecast whether price rise after k days (set to 1); forecast prices after k days (set to 2)
    days: set the value of k
Ouput
    return error
'''
def RF(data, forecastType, days): 
    # Smooth: Smooth the data
    # Input: d for data and a for alpha (the smoothing factor)
    # Output: smoothed data set
    def smooth(d, a):
        numData = len(d)
        SD = [0] * numData
        for idx, itr in enumerate(d):
            SD[idx] = a * itr + (1 - a) * SD[idx - 1] if idx != 0 else d[0]
        return SD
    def targetOne(d, numDays):
        tar = []
        for itr in range(len(d.Close) - numDays):
            tar.append(1) if d['Close'][itr + numDays] - d['Close'][itr] > 0 else tar.append(-1)
        d = d[ : len(d) - numDays]
        d = d.assign(Target = tar)
        return d
    def targetTwo(d, numDays):
        tar = []
        for itr in range(len(d.Close) - numDays):
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

    #Step 1: Smooth the data exponentially
    data.assign(Close = smooth(data.Close, 0.2))
    #Step 2: Set the target
    data = targetOne(data, days) if forecastType == 1 else targetTwo(data, days)
    #Step 3: Calculate the features
    features = ['RSI', 'SO', 'WIL', 'MACD', 'PRC', 'OBV']
    newData = FEATURE(data)
    CHECKDATA(newData)
    #Step 4: bagging/ bootstrap aggregating
    # print(newData)
    train, test = newData[ : (len(newData) * 7) // 10], newData[(len(newData) * 3) // 10 : ] #70% train, 30% test
    DS = bagging(train, 15, len(train) // 3) # Second parameter should be odd, otherwise we may receive some error
    #Step 5: Building the decision trees
    err = DT(DS, test, features, "Target", True, "Decision") if forecastType == 1 else DT(DS, test, features, "Target", True, "Prices")
    return err

'''
MainFn: main function
Input
    Does the calling of supporting functions
No Output
Note that we need to reload the data everytime we rerun the script
'''
def mainFn():
    TimePeriodOne = ['2002-12-03', '2003-06-03'] # 6 months, including one-month pre (for some lookback computation)
    TimePeriodTwo = ['2002-12-03', '2005-01-03'] # 2 years, including one-month pre (for some lookback computation)
    TimePeriodThree = ['2002-12-03', '2018-01-03'] # 15 years, including one-month pre (for some lookback computation)
    TimePeriodFour = ['1992-12-03', '2018-01-03'] # 25 years, including one-month pre (for some lookback computation)
    ticker = ['^GSPC', 'AAPL']
    SPShort, AAPLShort = yf.download('^GSPC', TimePeriodOne[0], TimePeriodOne[1]), yf.download('AAPL', TimePeriodOne[0], TimePeriodOne[1])
    SPMedium, AAPLMedium = yf.download('^GSPC', TimePeriodTwo[0], TimePeriodTwo[1]), yf.download('AAPL', TimePeriodTwo[0], TimePeriodTwo[1])
    SPLong, AAPLLong = yf.download('^GSPC', TimePeriodThree[0], TimePeriodThree[1]), yf.download('AAPL', TimePeriodThree[0], TimePeriodThree[1])
    SPExtraLong, AAPLExtraLong = yf.download('^GSPC', TimePeriodFour[0], TimePeriodFour[1]), yf.download('AAPL', TimePeriodFour[0], TimePeriodFour[1])
    if len(SPShort) < 1 or len(SPMedium) < 1 or len(SPLong) < 1 or len(SPExtraLong) < 1:
        raise ValueError('S&P Data Downloading had issues')
    if len(AAPLShort) < 1 or len(AAPLMedium) < 1 or len(AAPLLong) < 1 or len(AAPLExtraLong) < 1:
        raise ValueError('AAPL Data Downloading had issues')
    #optimalWinSize(SPMedium)
    #movingAverages(12, SPShort)
    #MAApplication(12, 100, SP) # Typical small window period is 12, 26 and typical long window period is 100, 200
    RF(SPMedium, 1, 5)
mainFn()