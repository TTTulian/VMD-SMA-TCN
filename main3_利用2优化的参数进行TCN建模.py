# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import matplotlib.pyplot as plt
import pandas as pd
from tcnutils import TCN,tcn_full_summary
from math import sqrt
from scipy.io import savemat,loadmat
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
tf.random.set_seed(0)
np.random.seed(0)
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus']=False

#时间窗口构造
def createXY(dataset,n_past):
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
            dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
            dataY.append(dataset[i,0])
    return np.array(dataX),np.array(dataY)

# In[] 加载数据
dataset = pd.read_excel("四种负荷低中高.xlsx")         #预测溶解氧
# features_Corrected_irradiance = [
#     '用电量','参数0','参数1','参数2','参数3','参数4','参数5','参数6'
# ]
features_Corrected_irradiance = [
    '1d低'
]
values = dataset[features_Corrected_irradiance].values
n_vars=values.shape[1]      #features_Corrected_irradiance中的个数

#
time_stemp=3  #时间步
## 确保所有数据是浮动的
values = values.astype('float32')

# 把数据集分为训练集和测试集
n_train_hours=int(0.8*values.shape[0])
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]

# 标准化
scaler = StandardScaler()
train = scaler.fit_transform(train)
test =  scaler.fit_transform(test)

#划分数据集
train_data, train_label=createXY(train,time_stemp)
test_data , test_label=createXY(test,time_stemp)
train_label=train_label.reshape(-1,1)

pop=loadmat('结果/SMA_para.mat')['best'].reshape(-1,)

#  加载参数
alpha= pop[0]  # 学习率
nb_filters0 = int(pop[1])#nb_filters
num_epochs = int(pop[2])#迭代次数
batch_size = int(pop[3])# batchsize

# In[]定义超参数
def tcn_model():
# 建立模型
    inputs = Input(shape=(train_data.shape[1], train_data.shape[2]))
    tcn = TCN(nb_filters=nb_filters0,  # 在卷积层中使用的过滤器数。可以是列表
              kernel_size=7,  # 在每个卷积层中使用的内核大小。
              nb_stacks=2,  # 要使用的残差块的堆栈数
              padding='same',
              dilations=[2 ** i for i in range(6)],  # 扩张列表。示例为：[1、2、4、8、16、32、64]。
              dropout_rate=0.15,  # 在0到1之间浮动。要下降的输入单位的分数。
              use_skip_connections=True,  # 是否要添加从输入到每个剩余块的跳过连接。
              kernel_initializer='Orthogonal',  # 内核权重矩阵（Conv1D）的初始化程序。
              activation='relu',  # 残差块中使用的激活函数 o = Activation(x + F(x)).
              name='tcn'  # 使用多个TCN时，要使用唯一的名称
              , return_sequences=False
              )(inputs)
    outputs = Dense(1)(tcn)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha),loss='mse')
    model.summary()#展示模型结构
    return model
model = tcn_model()#建立模型
history = model.fit(train_data, train_label, epochs=num_epochs, batch_size=batch_size, validation_data=(test_data , test_label), verbose=2,
                    shuffle=False)

# 绘制loss曲线
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# 作出预测 # 对测试结果进行反归一化
yhat = model.predict(test_data)
yhat = np.repeat(yhat,n_vars, axis=-1)
inv_yhat =scaler.inverse_transform(np.reshape(yhat,(len(yhat),n_vars)))[:,0]
y = np.repeat(test_label,n_vars, axis=-1)
inv_y =scaler.inverse_transform(np.reshape(y,(len(test_label),n_vars)))[:,0]

# #保存结果
# savemat('结果/sma-tcn_result.mat',{'true':inv_y,'pred':inv_yhat})

# In[]计算各种指标
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.7f' % rmse)
print('Test MAE: %.7f' % mean_absolute_error(inv_y, inv_yhat))
print('Test R2: %.7f' % r2_score(inv_y, inv_yhat))
ST = pd.DataFrame(inv_yhat)
ST .to_excel(excel_writer="1dd.xlsx")
# plot test_set result
plt.figure()
plt.plot(inv_y, c='r', label='real')
plt.plot(inv_yhat, c='b', label='pred')
plt.legend()
plt.xlabel('样本点')
plt.ylabel('值')
# plt.savefig('figure/SMA-TCN预测结果.jpg')
plt.show()

