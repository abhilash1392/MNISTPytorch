# Importing the libraries 
import numpy as np 
import idx2numpy
import torch 
from torch.utils.data import DataLoader,TensorDataset,SubsetRandomSampler

# Building a function for loading ANN 

def train_validation(Input):
    num_train =  len(Input)
    valid_size=0.2
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(num_train*valid_size))
    train_idx,valid_idx = indices[split:],indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    return train_sampler,valid_sampler

def annLoader(Input,label,train=True):
     
    Input = idx2numpy.convert_from_file(Input)
    label = idx2numpy.convert_from_file(label)
    Input = torch.tensor(Input/255.0,dtype=torch.float)
    label_1 = torch.tensor(label,dtype=torch.long)
    dataset = TensorDataset(Input,label_1)
    if train==True:
        train_sampler,valid_sampler = train_validation(Input)
        train_loader = DataLoader(dataset,batch_size=20,sampler=train_sampler)
        valid_loader = DataLoader(dataset,batch_size=20,sampler=valid_sampler)
        return train_loader,valid_loader 
    else:
        test_loader = DataLoader(dataset,batch_size=1)
        return test_loader


def cnnLoader(Input,label,train=True):
    Input = idx2numpy.convert_from_file(Input)
    label = idx2numpy.convert_from_file(label)
    Input = np.expand_dims(Input,1)
    Input = torch.tensor(Input/255.0,dtype=torch.float)
    label_1 = torch.tensor(label,dtype=torch.long)
    dataset = TensorDataset(Input,label_1)
    if train==True:
        train_sampler,valid_sampler = train_validation(Input)
        train_loader = DataLoader(dataset,batch_size=20,sampler=train_sampler)
        valid_loader = DataLoader(dataset,batch_size=20,sampler=valid_sampler)
        return train_loader,valid_loader 
    else:
        test_loader = DataLoader(dataset,batch_size=1)
        return test_loader
