import numpy as np
import pandas as pd
import sys
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn.modules.linear import Linear
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import TensorDataset

PERTUBTIME = 3
A_data_len = [67 ,100 ,100 ,100]
batch_size = 10
lr = 0.001
epochs = 500
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CreatDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x)
        self.y = torch.tensor(y)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        return self.x[index], self.y[index]



class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(10,40),
            nn.Tanh(),
            nn.Linear(40,40),
            nn.Tanh(),
            nn.Linear(40,40),
            nn.Tanh(),
            nn.Linear(40,40),
            nn.Tanh(),
            nn.Linear(40,40),
            nn.Tanh(),
            nn.Linear(40,40),
            nn.Tanh(),
            nn.Linear(40,40),
            nn.Tanh(),
            nn.Linear(40,40),
            nn.Tanh(),
            nn.Linear(40,40),
            nn.Tanh(),
            nn.Linear(40,40),
            nn.Tanh(),
            nn.Linear(40, 1)
        )
    def forward(self, x):
        out = self.layers(x)
        return out

if __name__ == '__main__':
    PERTUBTIME = 3
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

    print(x_train.shape)
    print(x_test.shape)
    torch_train_dataset = CreatDataset(x_train, y_train)
    torch_test_dataset = CreatDataset(x_test, y_test)
    loader_train = DataLoader(
            dataset=torch_train_dataset,
            batch_size=batch_size,
            num_workers=4)
    loader_test = DataLoader(
            dataset=torch_test_dataset,
            batch_size=batch_size,
            num_workers=4,
    )

    model = Net()
    model.to(dtype=float)
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ## Train
    loss_list=[]
    best_model = model
    best_score = sys.maxsize
    print(best_score)

    for epoch in tqdm(range(epochs)):
        model.train()
        running_loss=0.0
        for batch_idx, (data, target) in enumerate(loader_train):
            optimizer.zero_grad()
            #data, target = data.to(device), target.to(device)
            #data = data.view(-1 , 10)
            y_hat = model(data)
            target = target.to(dtype=float)
            loss = loss_func(y_hat, target)
            running_loss+=loss.item()
            loss.backward()
            optimizer.step()
            
        if running_loss < best_score:
            best_score = running_loss
            best_model = model

        print("Epoch_" + str(epoch) + "_loss: " + str(running_loss) )
        
        ##loss_list.append(running_loss)


    y = []
    y_hat = []
    mse=0
    mae=0
    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(enumerate(loader_test)):
            #data, target = data, target
            #data = data.view(-1 , 10)
            y_pred=best_model(data)
            mse += nn.MSELoss()(target, y_pred.flatten())
            mae += nn.L1Loss()(target, y_pred.flatten())
            for i in range(len(y_pred)):
                y_hat.append(y_pred[i])
                y.append(target[i])
        print("MSE: "+str(mse.item()))
        print("MAE: "+str(mae.item()))
    

    
    plt.plot(y, color="blue", label ="target")
    plt.plot(y_hat, color="red", label="predict")
    plt.legend()
    plt.show()
    #print(SelectedData.shape)'''