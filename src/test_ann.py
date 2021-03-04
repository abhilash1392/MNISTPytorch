import numpy as np 
import torch.nn as nn 
import torch  
from sklearn.metrics import accuracy_score
import idx2numpy 
import annNet 
import data_loader
import warnings
warnings.filterwarnings("ignore")
if __name__ == "__main__": 
    test_file = '../input/t10k-images-idx3-ubyte'
    test_labels = '../input/t10k-labels-idx1-ubyte'
    model = annNet.Net()
    model.load_state_dict(torch.load('../models/model_ann.pt'))
    test_loader = data_loader.annLoader(test_file,test_labels,train=False)
    true_labels = idx2numpy.convert_from_file(test_labels)
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    y_pred = []
    model.eval()
    for data,target in test_loader:
        output = model(data)
        loss = criterion(output,target)
        _,pred = torch.max(output,1)
        y_pred.append(pred)
        test_loss+=loss.item()*data.size(0)
    test_loss = test_loss/len(test_loader.sampler)
    print('Test Loss: {:.6f}'.format(test_loss))
    accuracy = accuracy_score(true_labels,y_pred)
    print('Model Accuracy : {:.2f}%'.format(accuracy*100))
