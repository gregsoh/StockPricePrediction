from EXP import EXP
from DT import DT
import fix_yahoo_finance as yf
import matplotlib.pyplot as plt
import pandas as pd

'''
MethodOne: Calling exponential moving average, changing
    1) Window size
    2) double or single moving average
Output: Graph of MSE against window size
'''
def methodOne():
    # Data processing
    s1, s2, s3 = '2003-01-01', '2008-01-01', '2013-01-01' #'s' stands for start
    e1, e2, e3 = '2008-01-01', '2013-01-01', '2018-01-01' #'e' stands for end
    SP1, SP2, SP3 = yf.download('^GSPC', s1, e1), yf.download('^GSPC', s2, e2), yf.download('^GSPC', s3, e3) #S&P 500
    # Main function
    winSize = [(itr + 1) * 25 for itr in range(20)]
    val1, val2 = [], []
    for itr in winSize:
        val1.append((EXP(SP1.Close, itr, False, False) + EXP(SP2.Close, itr, False, False) + EXP(SP3.Close, itr, False, False)) / 3)
        val2.append((EXP(SP1.Close, itr, True, False) + EXP(SP2.Close, itr, True, False) + EXP(SP3.Close, itr, True, False)) / 3)
    plt.scatter(winSize, val1), plt.scatter(winSize, val2)
    plt.xlabel('Window Size'), plt.ylabel('Mean Square Error')
    plt.title('Exponential Moving Average')
    plt.legend(['S&P EMA', 'S&P DEMA'], loc='upper right')
    plt.savefig('EMA.png')
    plt.show()

'''
methodTwo: Decision Trees
Output:
'''
def methodTwo():
    # Data processing
    s1, e1 = '2003-01-03', '2018-12-14'
    SP = yf.download('^GSPC', s1, e1)
    df = pd.read_excel('ED.xlsx', sheet_name='DATA', header = 12, index_col = 0)
    merged = pd.merge(df.dropna(thresh = 3), SP, left_index = True, right_index = True)
    merged = merged.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis = 1)
    # rounding off values to 2 decimal points (has "binning" effect as we want our tree size to be manageable)
    merged = merged.round(2)
    # test: 2003-01-02 to 2013-01-02 (excl.) and train: 2013-01-02 to 2018-12-13
    train, test = merged[:2496], merged[2496:] #70% train, 30% test
    features = ['DTWEXM', 'DEXCHUS', 'T10YIE']
    return DT(train, test, features)
methodOne()
#print(methodTwo())

