import torch.nn as nn
import torch

class ConvModel(nn.Module):
    def __init__(self,shape_in):
        super(ConvModel, self).__init__()

        self.conv1 = nn.Conv2d(shape_in[-3], 32, kernel_size=(3, 3), padding=(1, 1))

        self.conv2 = nn.Conv2d(32, 32, kernel_size = (3,3), stride = 2, padding = (1,1))

        self.conv3 = nn.Conv2d(32, 64, kernel_size = (3,3), padding = (1,1))

        self.conv4 = nn.Conv2d(64, 64, kernel_size = (3,3), stride = 2, padding = (1,1))

        num_dense = 69696
        self.fc1 = nn.Linear(num_dense, 1)

    def forward(self,x):

        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = self.conv4(x)
        x = nn.ReLU()(x)
        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = nn.Sigmoid()(x)
        return x

    def enumerate_parameters(self):
        for layer in self.children():
            print(layer)

    def freeze(self,num_last):
        num_layers = len(list(self.children())) - num_last
        for i,layer in enumerate(self.children()):
            if i < num_layers:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                for param in layer.parameters():
                    param.requires_grad = True
    
    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

