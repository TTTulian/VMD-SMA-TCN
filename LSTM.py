# -*- coding = utf-8 -*-
# @Time : 2022/7/12 15:46
# @Author : LH
import numpy as np
import pandas as pd
from pandas import read_csv, read_excel
from pandas import datetime
from pandas import concat
from pandas import DataFrame
from pandas import Series
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
import numpy

from pandas import read_csv
from pandas import datetime
from pandas import concat
from pandas import DataFrame
from pandas import Series
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
import numpy
from sklearn.metrics import r2_score
from tcn.tcn import TCN


# 读取时间数据的格式化

def MAPE(pred,true):
    return np.mean(np.abs((pred - true)/true))

def RMSE(pred, true):
    return np.sqrt(np.mean(np.square(pred - true)))


def parser(x):
    return datetime.strftime(x, '%Y-%m-%d')


# 转换成有监督数据
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]  # 数据滑动一格，作为input，df原数据为output
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df


# 差分
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)


# 逆差分
def inverse_difference(history, yhat, interval=1):  # 历史数据，预测数据，差分间隔
    return yhat + history[-interval]


# 缩放
def scale(train, test):
    # 根据训练数据建立缩放器
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # 转换train data
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # 转换test data
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled





# 逆缩放
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = numpy.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]


# fit LSTM来训练数据
def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    # 添加LSTM层
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))  # 输出层1个node
    # 编译，损失函数mse+优化算法adam
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        # 按照batch_size，一次读取batch_size个数据
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
    return model


# 1步长预测
def forcast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0, 0]


# 加载数据
series = read_excel('lstm.xlsx', header=0, parse_dates=[0], index_col=0, squeeze=True,
                  date_parser=parser)

# 让数据变成稳定的
raw_values = series.values
diff_values = difference(raw_values, 1)

# 变成有监督数据
supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values

# 数据拆分：训练数据、测试数据
train, test = supervised_values[0:11520], supervised_values[11520:14400]#0:-12是去掉最后12数；-12：是剩最后12

# 数据缩放
scaler, train_scaled, test_scaled = scale(train, test)

# fit 模型#######################################################################################
lstm_model = fit_lstm(train_scaled, 1, 100, 4)  # 训练数据，batch_size，epoche次数, 神经元个数
# 预测
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
lstm_model.predict(train_reshaped, batch_size=1)

# 测试数据的前向验证
predictions = list()
for i in range(len(test_scaled)):
    # 1步长预测
    X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
    yhat = forcast_lstm(lstm_model, 1, X)
    # 逆缩放
    yhat = invert_scale(scaler, X, yhat)
    # 逆差分
    yhat = inverse_difference(raw_values, yhat, len(test_scaled) + 1 - i)
    predictions.append(yhat)
    # expected = raw_values[len(train) + i + 1]
    # print('Moth=%d, Predicted=%f, Expected=%f' % (i + 1, yhat, expected))

# predown = pd.DataFrame(predictions)
# predown.to_excel(excel_writer="L-preup.xlsx")
# 性能报告
rmse = sqrt(mean_squared_error(raw_values[11521:14400], predictions))
# print('RMSE ', RMSE(scaled_prediction, scaled_test_label))
print('MAPE: ', MAPE(predictions, raw_values[11521:14400]))
print('R2:', r2_score(raw_values[11521:14400],predictions))
print('RMSE:%.3f' % rmse)
LSTM = pd.DataFrame(predictions)
LSTM.to_excel(excel_writer="LSTM-4下.xlsx")
## 绘图
pyplot.plot(raw_values[11521:14400],color = 'c')
pyplot.plot(predictions,color = 'r')
pyplot.show()
