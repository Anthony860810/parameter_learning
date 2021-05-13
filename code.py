import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
import matplotlib
import matplotlib.pyplot as plt

PERTUBTIME = 2
A_data_len = [67 ,100 ,100 ,100]
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    "learning_rate":0.1,
    "num_leaves": 255,
    "max_bin": 1024,
    'objective': 'regression',
    'metric': {'l2'},
    'verbose': 0
}



def DataGenerate():
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
            
            #training_data = np.append(training_data,feat_data_reglab[i,:])
            #training_data = np.append(training_data,0)
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
    #print(training_data,training_data.shape)
    return training_data
def SelectFeatures(dataset):
    pca = PCA(n_components=10)
    newdata = pca.fit_transform(dataset)
    return newdata

def Split(x,y):
    return  train_test_split(x, y, test_size=0.1, random_state=0, shuffle=True)

if __name__ == '__main__':
    data = DataGenerate()
    y = data[:,-2]
    print(data.shape)
    ## 後三個都是evaluation後三個都是evaluation
    x= data[:-3]
    print(x.shape)
    #print(data[:,-4])
    x = SelectFeatures(data)
    X_train, X_test, y_train, y_test =Split(x,y)
    X_train, X_eval, y_train, y_eval = Split(X_train,y_train)
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_eval, y_eval, reference=lgb_train)
    model = lgb.train(params, lgb_train, valid_sets=lgb_eval, early_stopping_rounds=5)
    y_pred = model.predict(X_test)
    #model = LinearRegression().fit(X_train,y_train)
    y_pred = model.predict(X_test)
    plt.plot(y_test, color="blue")
    plt.plot(y_pred, color="red")
    plt.show()
    #print(SelectedData.shape)