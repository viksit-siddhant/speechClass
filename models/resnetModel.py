from torch import nn
from torchvision.models import resnet34

class ResNetModel(nn.Module):
    def __init__(self):
        super(ResNetModel, self).__init__()
        self.resnet = resnet34(weights=None)
        self.fc2 = nn.Linear(1000,1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):
        x = self.resnet(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
