import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.modules import padding 


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,16,3,padding=1)
        self.conv2 = nn.Conv2d(16,32,3,padding=1)
        self.conv3 = nn.Conv2d(32,64,3,padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(576,256)
        self.fc2 = nn.Linear(256,10)
        self.dropout = nn.Dropout(0.2)
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))        
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.shape[0],-1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.softmax(self.fc2(x))
        return x 


if __name__=="__main__":
    model = Net()
    print(model)




