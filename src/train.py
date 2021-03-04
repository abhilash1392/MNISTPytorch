import numpy as np
import annNet
import torch.nn as nn 
import torch.optim as optim
import torch
# import cnnNet
import data_loader 
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")
if __name__=="__main__":
    train_images = '../input/train-images-idx3-ubyte'
    train_labels = '../input/train-labels-idx1-ubyte'
    train_loader,valid_loader = data_loader.annLoader(train_images,train_labels,train=True)
    model=annNet.Net()
    criterion= nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),lr=0.01)
    n_epochs = 30 
    validation_loss_min = np.Inf 
    for epoch in range(1,n_epochs+1):
        train_loss=0.0
        valid_loss = 0.0
        model.train()
        for data,target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output,target)
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()*data.size(0)
        
        model.eval()
        for data,target in valid_loader:
            output = model(data)
            loss = criterion(output,target)
            valid_loss += loss.item()*data.size(0)

        train_loss = train_loss/len(train_loader.sampler)
        valid_loss = valid_loss/len(valid_loader.sampler)

        print('Epoch : {} Train Loss : {:.6f}  Valid Loss : {:.6f}'.format(epoch,train_loss,valid_loss))

        if valid_loss<=validation_loss_min:
            print('Valid loss has decreased {:.6f} --->>> {:.6f}.  Saving Model'.format(validation_loss_min,valid_loss))
            torch.save(model.state_dict(),'../models/model_ann.pt')
            validation_loss_min=valid_loss


        



