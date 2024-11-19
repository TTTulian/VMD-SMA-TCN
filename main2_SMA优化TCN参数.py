# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
import pandas as pd
from tcnutils import TCN,tcn_full_summary
import math
import random
import copy
from math import sqrt
from scipy.io import savemat,loadmat
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
n_train_hours=int(0.9*values.shape[0])
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


'''
进行适应度计算,以均方差为适应度函数，目的是找到一组超参数 使得网络的误差最小
'''
#  优化四个参数
def fun(pop, P, T, Pt, Tt):
    tf.random.set_seed(0)
    alpha = pop[0]  # 学习率
    nb_filters0 = int(pop[1])  # 第一隐含层神经元
    num_epochs = int(pop[2])#迭代次数
    batch_size = int(pop[3])# batchsize

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
              )(inputs)
    outputs = Dense(1)(tcn)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha), loss='mse')
    model.summary()  # 展示模型结构
    model.fit(train_data, train_label, epochs=num_epochs, batch_size=batch_size, validation_data=(test_data , test_label), verbose=2,
              shuffle=False)


    test_pred = model.predict(test_data)
    F2 = np.mean(np.square((test_pred - test_label)))
    return F2



'''边界检查函数'''
def BorderCheck(pop,lb,ub):
    pop=pop.flatten()
    lb=lb.flatten()
    ub=ub.flatten()
    # 防止跳出范围,除学习率之外 其他的都是整数
    pop=[int(pop[i]) if i>0 else pop[i] for i in range(lb.shape[0])]
    for i in range(len(lb)):
        if pop[i]>ub[i] or pop[i]<lb[i]:
            if i==0:
                pop[i] = (ub[i]-lb[i])*np.random.rand()+lb[i]
            else:
                pop[i] = np.random.randint(lb[i],ub[i])
    return pop

''' 种群初始化函数 '''
def initial(pop, dim, ub, lb):
    X = np.zeros([pop, dim])
    for i in range(pop):
        for j in range(dim):
            X[i, j] = random.random() * (ub[j] - lb[j]) + lb[j]

    return X, lb, ub
'''计算适应度函数'''
def CaculateFitness(X, fun,P,T,Pt,Tt):
    pop = X.shape[0]
    fitness = np.zeros([pop, 1])
    for i in range(pop):
        fitness[i] = fun(X[i, :],P,T,Pt,Tt)
    return fitness

'''适应度排序'''
def SortFitness(Fit):
    fitness = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return fitness, index


'''根据适应度对位置进行排序'''
def SortPosition(X, index):
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i, :] = X[index[i], :]
    return Xnew
    return Xnew


'''黏菌优化算法'''
def SMA(P,T,Pt,Tt):
    pop = 10     #pop种群数量     #MaxIter和pop这两个参数设置的越大  相对来说寻优出来适应度越好效果越好  ，但是算法运行花的时间就越多
    MaxIter = 10  #MaxIter迭代次数
    # 第一个是学习率[0.001 0.01]
    # 第二个是神经元个数[10-100]
    lb = np.array([0.001, 1,10,15]).reshape(-1, 1)
    ub = np.array([0.01, 15,40,70]).reshape(-1, 1)
    dim=4  # 搜索维度   寻优了几个参数就是几
    z = 0.03  # 位置更新参数

    X, lb, ub = initial(pop, dim, ub, lb)  # 初始化种群
    fitness = CaculateFitness(X, fun, P, T, Pt, Tt)  # 计算适应度值
    fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序
    X = SortPosition(X, sortIndex)  # 种群排序
    GbestScore = copy.copy(fitness[0])
    GbestPositon = copy.copy(X[0, :])
    result = np.zeros([MaxIter, dim])
    Curve = np.zeros([MaxIter, 1])
    W = np.zeros([pop, dim])  # 权重W矩阵
    for t in range(MaxIter):
        worstFitness = fitness[-1]
        bestFitness = fitness[0]
        S = bestFitness - worstFitness + 10E-8  # 当前最优适应度于最差适应度的差值，10E-8为极小值，避免分母为0；
        for i in range(pop):
            if i < pop / 2:  # 适应度排前一半的W计算
                W[i, :] = 1 + np.random.random([1, dim]) * np.log10((bestFitness - fitness[i]) / (S) + 1)
            else:  # 适应度排后一半的W计算
                W[i, :] = 1 - np.random.random([1, dim]) * np.log10((bestFitness - fitness[i]) / (S) + 1)
        # 惯性因子a,b
        tt = -(t / MaxIter) + 1
        if tt != -1 and tt != 1:
            a = math.atanh(tt)
        else:
            a = 1
        b = 1 - t / MaxIter
        # 位置更新
        for i in range(pop):
            if np.random.random() < z:
                X[i, :] = (ub.T - lb.T) * np.random.random([1, dim]) + lb.T
            else:
                p = np.tanh(abs(fitness[i] - GbestScore))
                vb = 2 * a * np.random.random([1, dim]) - a
                vc = 2 * b * np.random.random([1, dim]) - b
                for j in range(dim):
                    r = np.random.random()
                    A = np.random.randint(pop)
                    B = np.random.randint(pop)
                    if r < p:
                        X[i, j] = GbestPositon[j] + vb[0, j] * (W[i, j] * X[A, j] - X[B, j])
                    else:
                        X[i, j] = vc[0, j] * X[i, j]

        for i in range(pop):
            X[i, :] = BorderCheck(X[i, :], lb, ub)
        fitness = CaculateFitness(X, fun, P, T, Pt, Tt)  # 计算适应度值
        fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序
        X = SortPosition(X, sortIndex)  # 种群排序
        if (fitness[0] <= GbestScore):  # 更新全局最优
            GbestScore = copy.copy(fitness[0])
            GbestPositon = copy.copy(X[0, :])
        Curve[t] = GbestScore
        result[t, :] = GbestPositon
        print(GbestPositon.shape)
    return GbestPositon,Curve,result


best,trace,result=SMA(train_data,train_label,test_data,test_label)
savemat('结果/SMA_para.mat',{'trace':trace,'best':best,'result':result})
print("最优学习率、神经元的参数分别为：",[int(best[i]) if i>0 else best[i] for i in range(len(best))])

#迭代次数适应度函数曲线
mpl.rcParams['font.sans-serif'] = ['SimHei']   #显示中文
mpl.rcParams['axes.unicode_minus']=False       #显示负号
plt.figure(figsize=(6,4),dpi=500)
plt.plot(trace,'r',linestyle="--",linewidth=0.5)
plt.xticks(list(range(0,31, 5)))
plt.xlabel('迭代次数',fontsize=10)
plt.ylabel('适应度值',fontsize=10)
plt.show()
trace = pd.DataFrame(trace)
trace.to_excel(excel_writer="trace.xlsx")