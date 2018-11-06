import torch
import torchvision
from torch import nn
from torch import optim
from torchvision import models
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from torchvision.utils import save_image
from torch.optim import lr_scheduler
import numpy as np
from sklearn.model_selection import train_test_split
import os
from collections import namedtuple
import argparse
from shutil import copyfile
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
#from lstm_model_classification import LSTM
from PIL import *
torch.backends.cudnn.enabled=False
import pandas as pd
import torch.autograd as autograd

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

#LSTM model class
class LSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, output_size):

        super(LSTM, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=3, batch_first = True)

        self.hidden2out = nn.Linear(hidden_dim, output_size)
        self.softmax = nn.LogSoftmax()

        self.dropout_layer = nn.Dropout(p=0.2)


    def init_hidden(self, batch_size):
        return(autograd.Variable(torch.randn(3, batch_size, self.hidden_dim)).cuda(),
                        autograd.Variable(torch.randn(3, batch_size, self.hidden_dim)).cuda())


    def forward(self, batch):

        self.hidden = self.init_hidden(batch.size(0))

        #embeds = self.embedding(batch)
        #print(batch, self.hidden)
        #packed_input = pack_padded_sequence(embeds, lengths)
        outputs, (ht, ct) = self.lstm(batch, self.hidden)

        # map from the original feature space to our features of interest with a linear layer (dropout for regularization)
        output = self.dropout_layer(ht[-1])
        output = self.hidden2out(output)
        #output[:,-4] = self.softmax(output[:,-4])
        return output, outputs


 #Dataloader for the LSTM. (advanced give mean of the the current hour as well
 #Features of interst are density of the classes and number of total objects per hour (these are predicted with the LSTM)
 #Our Input space consists of additional features as weather information, holiday information, weekday (0-6), hour etc.
class Features(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dic, datas, dic_mean):
        """
			 Dataloader for the LSTM. (advanced give mean of the the current hour as well
			 Features of interst are density of the classes and number of total objects per hour (these are predicted with the LSTM)
			 Our Input space consists of additional features as weather information, holiday information, weekday (0-6), hour etc.
        """
        self.info = dic
        self.all = datas
        self.count, self.q_1, self.q_2, self.q_3 = dic_mean
        #save mean and std to renormalize data afterwards (only for number of obejcts)
        self.mean1 = np.mean(self.all[:,-5],axis=0)
        self.std1 = np.std(self.all[:,-5],axis=0)
        print(self.mean1, self.std1)
        #normalize data for using it (dont normalize traffic density because it is already normlized)
        self.new = np.copy(datas[:,1:])
        self.day = np.copy(datas[:,2])
        self.new[:,:-4] = (self.all[:,1:-4] - np.mean(self.all[:,1:-4].astype(float), axis=0))/(np.std(self.all[:,1:-4].astype(float), axis=0)+1e-4)
        
        
    def __len__(self):
        return (len(self.info))

    def __getitem__(self, idx):
        
        #read out index array to know which array we use
        new_idx = self.info[idx]
        if new_idx < 25: #solve issue if idx is smaller than vocab size
            new_idx +=25
        feature = self.new[new_idx-25:new_idx,:] #load 25 frames, predict with the last 25 the next 24
        new_features = np.zeros((25,4))
        for i in range(25):
                curr_day = self.day[new_idx-25]
                new_features[i,0] = self.count[int((new_idx-1)%24), int(curr_day-1)]
                new_features[i,1] = self.q_1[int((new_idx-1)%24), int(curr_day-1)]
                new_features[i,2] = self.q_2[int((new_idx-1)%24), int(curr_day-1)]
                new_features[i,3] = self.q_3[int((new_idx-1)%24), int(curr_day-1)]
        feature = np.concatenate((new_features, feature),axis=1)        
        sample = {'features': feature, 'std1':self.std1, 'mean1':self.mean1, 'bin':new_idx}

        return sample

#train function

def train(model, batch_size, num_epochs, lr, path_to_save_imgs):

    if not os.path.exists(path_to_save_imgs):
        os.mkdir(path_to_save_imgs)

    data = np.load('./data/pandas_wide_extended.npy', encoding="latin1")
    dic1 = np.load('./data/p1_mat.npy', encoding="latin1")
    dic2 = np.load('./data/p2_mat.npy', encoding="latin1")
    dic3 = np.load('./data/p3_mat.npy', encoding="latin1")
    dic4 = np.load('./data/p4_mat.npy', encoding="latin1")
    dic = (dic1, dic2, dic3, dic4)
    #train, test, _, _ = train_test_split(np.linspace(0,len(data)-1,len(data),dtype=int), np.zeros((len(data))), test_size=0.20, random_state=42)
    
    #we split our dataset in test and train. Therefor we used the first 80% of our data for training and the last 20% for testing. 
    train = np.arange(0,len(train))
    key_dataset = Features(train, data, dic)
    dataloader_train = DataLoader(key_dataset, batch_size, shuffle=True)
    test = np.arange(len(train),len(train)+len(test))
    key_dataset = Features(test, data, dic)
    dataloader_test = DataLoader(key_dataset, batch_size, shuffle=False)
    
    #if torch.cuda.is_available():
    model.cuda()

    #As Objective Function we used a L2 error
    MSE = nn.MSELoss(size_average=True)
    optimizer = optim.Adam(model.parameters(), lr)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        #change learning with number of epochs
        if epoch == 100:
            lr *= 0.1
        if epoch == 50:
            lr *= 0.1
        for batch_idx, data in enumerate(dataloader_train):
            features = data['features']
            features = Variable(features).float()
            if torch.cuda.is_available():
                features = features.cuda()
            optimizer.zero_grad()
            output, _ = model(features[:,:-1,:])
            loss = MSE(output, features[:,-1,-5:])
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        av_loss_train = train_loss / len(dataloader_train)
        print('====> Epoch: {} Average train loss: {:.4f}'.format(epoch, av_loss_train))
        if (epoch+1) % 5 == 0:
            test_model(model, dataloader_test, MSE, epoch)


def test_model(model, dataloader_test, MSE, epoch):    
    model.eval()
    test_loss = 0; start = 0
    bins = np.zeros((len(dataloader_test.dataset),))
    forecast = np.zeros((len(dataloader_test.dataset),5))
    for batch_idx, data in enumerate(dataloader_test):
        features = data['features']
        mean, std = data['mean1'], data['std1']
        features = Variable(features).float()
        if torch.cuda.is_available():
            features = features.cuda()
        output, outputs = model(features[:,:-1,:])
        bins[start:start+outputs.size(0)] = data['bin']
        loss = MSE(output, features[:,-1,-5:])
        out = output[:,:].cpu().detach().numpy()
        out[:,0] = out[:,0] * std + mean
        forecast[start:start+outputs.size(0),:] = out
        start = (batch_idx+1)*outputs.size(0)
        test_loss += loss.item()
    if epoch == 199 or epoch <10:
        np.save('forecasted.npy', forecast)
        np.save('bins.npy', bins)
    av_loss_test = test_loss / len(dataloader_test)
    print('====> Average test loss: {:.4f}'.format(av_loss_test))

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
	#torch.cuda.set_device(1)
	model = LSTM(15, 15, 5)
	path_to_save_imgs = 'model/'
	batch_size = 200; num_epochs = 200; lr = 1e-2
	train(model, batch_size, num_epochs, lr, path_to_save_imgs)