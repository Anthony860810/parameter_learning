import re
import numpy as np
import pandas as pd
import sys
import os
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

loss = []
for i in range(500):
    print("Epoch: ",i)
    PERTUBTIME = 2
    A_data_len = [67 ,100 ,100 ,100]
    feat_data = pd.read_csv("TSfeatA1Benchmark.csv",index_col = 0)
    lab_data = pd.read_csv("A1_Label.csv",index_col = 0)
    feat_data = feat_data.values
    lab_data = lab_data.values
    mean_thre = np.reshape(((lab_data[:,-1] + lab_data[:,-2])/2),(-1,1))
    lab_data = np.append(lab_data,mean_thre,axis = 1)
    #Prepare for final training data
    training_data = np.array([])
    feat_data_reglab = np.append(feat_data,mean_thre,axis = 1)
    scaler_Train = MinMaxScaler(feature_range=(-1, 1))
    feat_data_reglab = scaler_Train.fit_transform(feat_data_reglab,(-1,1))

    for i in range(A_data_len[0]):

        #for unhandling data
        if lab_data[i,0] == 0:
            
            training_data = np.append(training_data,feat_data_reglab[i,:])
            training_data = np.append(training_data,0)
            continue

        #for data who didn't utilize X'(t)
        elif lab_data[i,1] == 1:

            buffer = np.array([])

            for j in range(int(3*PERTUBTIME)):

                per_ele = np.random.randint(-1,1, size=feat_data_reglab.shape[1])
                final_input = feat_data_reglab[i,:] + (feat_data_reglab[i,:]*per_ele/100)
                buffer = np.append(buffer,final_input)
                buffer = np.append(buffer,1)
            
            buffer = np.reshape(buffer,(-1,feat_data_reglab.shape[1]+1))
            training_data = np.append(training_data,buffer)

        #for optsaliency map
        else:

            buffer = np.array([])

            for j in range(PERTUBTIME):

                per_ele = np.random.randint(-1,1, size=feat_data_reglab.shape[1])
                final_input = feat_data_reglab[i,:] + (feat_data_reglab[i,:]*per_ele/100)
                buffer = np.append(buffer,final_input)
                buffer = np.append(buffer,0)
            
            buffer = np.reshape(buffer,(-1,feat_data_reglab.shape[1]+1))
            training_data = np.append(training_data,buffer)
    #[:,-1] ==> label of Xt (for classify),[:,-2] ==> label of threshold (for regression)
    training_data = np.reshape(training_data,(-1,feat_data_reglab.shape[1]+1))

    #==============
    #start training ---> PCA+DNN
    EPOCHS = 500
    BATCHSIZE = 10
    HIDDEN_NEURON_SCALE = 4
    PCA_COMPONENTS = 10

    pca = PCA(n_components=PCA_COMPONENTS)
    pca.fit(training_data[:,:-2])
    training_data_pca = pca.transform(training_data[:,:-2])
    stick_reglab = np.reshape(training_data[:,-2],(-1,1))
    training_data_pca = np.append(training_data_pca,stick_reglab,axis = 1)
    scaler_DNN = MinMaxScaler(feature_range=(-1, 1))
    training_data_pca = scaler_DNN.fit_transform(training_data_pca,(-1,1))

    #split in to the train and test set
    row = round(0.9 * training_data_pca.shape[0])
    np.random.shuffle(training_data_pca)
    train = training_data_pca[:int(row), :]
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = training_data_pca[int(row):, :-1]
    y_test = training_data_pca[int(row):, -1]

    #print(x_train.shape[1])
    #print(x_test.shape)

    model = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, random_state =7, nthread = -1).fit(x_train,y_train)
    predicted = model.predict(x_test)
    predicted = predicted.reshape(len(predicted),1)

    #print(predicted.shape)
    inverse_x = np.concatenate((x_test, predicted), axis=1)
    inverse_x = scaler_DNN.inverse_transform(inverse_x)
    inverse_x = inverse_x[:,-1]
    inverse_x = np.reshape(inverse_x,(-1,1))
    inverse_x = np.concatenate((training_data[-len(inverse_x):,:-2], inverse_x), axis=1)
    inverse_x = scaler_Train.inverse_transform(inverse_x)
    inverse_x = inverse_x[:,-1]

    y_test = y_test.reshape((len(y_test), 1))
    inverse_y = np.concatenate((x_test, y_test), axis=1)
    inverse_y = scaler_DNN.inverse_transform(inverse_y)
    inverse_y = inverse_y[:,-1]
    inverse_y = np.reshape(inverse_y,(-1,1))
    inverse_y = np.concatenate((training_data[-len(inverse_y):,:-2], inverse_y), axis=1)
    inverse_y = scaler_Train.inverse_transform(inverse_y)
    inverse_y = inverse_y[:,-1]
    loss.append(mean_squared_error(inverse_y,inverse_x))
    print('mse = ' , mean_squared_error(inverse_y,inverse_x))
loss = np.array(loss)
print(loss)
print("Mean: ",np.mean(loss))
print("母體標準差: ", np.std(loss, ddof=0))
print("樣本標準差: ", np.std(loss, ddof=1))
# fig = plt.figure(facecolor='white')
# ax = fig.add_subplot(111)
# ax.plot(inverse_y, label='True Data')
# plt.plot(inverse_x, label='Prediction')
# plt.legend()
# plt.show()