# -*- coding = utf-8 -*-
# @Time : 2022/6/28 16:38
# @Author : LH
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tcn.tcn import TCN
from tensorflow import keras

from sklearn.metrics import r2_score


window_size = 10  # 窗口大小
batch_size = 2  # 训练批次大小
epochs = 100  # 训练epoch
filter_nums = 10  # filter数量
kernel_size = 4  # kernel大小
o1 = pd.read_excel('tcn.xlsx',skiprows=1)
o1=o1.values[:,1]


def get_dataset():
    df = pd.read_excel('tcn.xlsx')
    # df = pd.read_csv("SXVMDban-2.csv")
    scaler = MinMaxScaler()
    open_arr = scaler.fit_transform(df['Open'].values.reshape(-1, 1)).reshape(-1)
    X = np.zeros(shape=(len(open_arr) - window_size, window_size))
    label = np.zeros(shape=(len(open_arr) - window_size))
    for i in range(len(open_arr) - window_size):
        X[i, :] = open_arr[i:i + window_size]
        label[i] = open_arr[i + window_size]
    train_X = X[0:13700]
    train_label = label[0:13700]
    test_X = X[13700:14400]
    test_label = label[13700:14400]
    return train_X, train_label, test_X, test_label, scaler


def RMSE(pred, true):
    return np.sqrt(np.mean(np.square(pred - true)))
def MAPE(pred,true):
    return np.mean(np.abs((pred - true)/true))


def plot(pred, true):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(len(pred)), pred,color = 'r')
    # ax.plot(range(len(true)), true,color = 'k')
    ax.plot(range(len(o1[13700:14400])), o1[13700:14400],color = 'k')
    plt.show()


def build_model():
    train_X, train_label, test_X, test_label, scaler = get_dataset()
    model = keras.models.Sequential([
        keras.layers.Input(shape=(window_size, 1)),
        TCN(nb_filters=filter_nums,  # 滤波器的个数，类比于units
            kernel_size=kernel_size,  # 卷积核的大小
            dilations=[1, 2, 4, 8]),  # 空洞因子
        keras.layers.Dense(units=1, activation='relu')
    ])
    model.summary()
    model.compile(optimizer='adam', loss='mae', metrics=['mae'])
    model.fit(train_X, train_label, validation_split=0.2, epochs=epochs)

    model.evaluate(test_X, test_label)
    prediction = model.predict(test_X)
    scaled_prediction = scaler.inverse_transform(prediction.reshape(-1, 1)).reshape(-1)
    scaled_test_label = scaler.inverse_transform(test_label.reshape(-1, 1)).reshape(-1)
    print('MAPE ', MAPE(scaled_prediction, scaled_test_label))
    print('R2', r2_score(scaled_test_label,scaled_prediction))
    print('RMSE ', RMSE(scaled_prediction, scaled_test_label))
    plot(scaled_prediction, scaled_test_label)
    preup = pd.DataFrame(scaled_prediction)
    preup.to_excel(excel_writer="TCN-1ug.xlsx")
    # predown = pd.DataFrame(scaled_prediction)
    # predown.to_excel(excel_writer="predown.xlsx")



if __name__ == '__main__':
    build_model()

