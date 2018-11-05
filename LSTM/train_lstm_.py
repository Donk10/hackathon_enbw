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
import os
from collections import namedtuple
import argparse
from shutil import copyfile
import glob
from tqdm import tqdm
#from lstm_model_classification import LSTM
from PIL import *

class Features(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, npy_file):
        """
        Args:
            npy_file (string): Path to the npy dictionary with information about fc6.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.info = np.load(npy_file, encoding="latin1")

    def __len__(self):
        return (len(self.info))

    def __getitem__(self, idx):

        if idx < 25:
            idx += 25
        feature = self.info[idx-25:idx]

        sample = {'features': feature}

        return sample


def test(model, dataloader_test, MSE):    
    model.eval()
    test_loss = 0
    for batch_idx, data in enumerate(dataloader_test):
        features, _ = data
        features = Variable(features)
        if torch.cuda.is_available():
            features = features.cuda()
        prediction = model(features[:-1])
        loss = MSE(prediction, features[-1])
        test_loss += loss.item()

    av_loss_test = test_loss / len(dataloader_test.dataset)
    print('====> Average test loss: {:.4f}'.format(av_loss_test))

def train(model, batch_size, num_epochs, lr, path_to_save_imgs):

    if not os.path.exists(path_to_save_imgs):
        os.mkdir(path_to_save_imgs)

    data_dir = ''
    key_dataset = Features(npy_file=data_dir+'features.npy')
    dataloader_train = DataLoader(key_dataset, batch_size, shuffle=True)

    model.cuda()

    MSE = nn.MSELoss(size_average=False)
    optimizer = optim.Adam(model.parameters(), lr)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_idx, data in enumerate(dataloader_train):
            features = data['features']
            features = Variable(features)
            if torch.cuda.is_available():
                features = features.cuda()
            optimizer.zero_grad()
            outputs, (ht, ct) = model(features[:,:-1,:])
            #print(';;;;;;;;;;;;;;;;;;;;;;;;;', features[:,-1,:].size())
            loss = MSE(outputs, features[:,-1,:])
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.3f}\t'.format(
                    epoch,
                    batch_idx * len(features),
                    len(dataloader_train.dataset), 100. * batch_idx / len(dataloader_train),
                    loss.data[0] / len(features)))
        av_loss_train = train_loss / len(dataloader_train.dataset)
        print('====> Epoch: {} Average train loss: {:.4f}'.format(epoch, av_loss_train))
        lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, threshold=0.01)
        if (epoch+1) % 5 == 0:
            test(model, dataloader_test, MSE)
        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), path_to_save_imgs+'/cae_bce.pth')

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    #model = LSTM(embedding_dim = 10 , hidden_dim = num_dim , output_size = num_dim)
    model = nn.LSTM(10, 10, num_layers=1)
    #model.load_state_dict(torch.load('/export/home/mdorkenw/HumanGaitDataset/VAE/VAE_serpate_new'+'/cae_bce.pth'))
    path_to_save_imgs = 'model/'
    batch_size = 100; num_epochs = 100; lr = 1e-3
    train(model, batch_size, num_epochs, lr, path_to_save_imgs)