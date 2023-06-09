import torch
from torchvision.models import alexnet
from torchvision.transforms.functional import to_tensor, resize

from sklearn.svm import LinearSVC

class Model:
    def __init__(self):
        self.model = alexnet(pretrained=True)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.features = []
        self.model.classifier[-2].register_forward_hook(self.get_features('fc7'))

        for layer in self.model.children():
            for param in layer.parameters():
                param.requires_grad = False
        
        self.svm = LinearSVC(C=10e-3)
        
    def get_features(self,name):
        def hook(model, input, output):
            self.features.append(output.detach())
        return hook

    def get_data(self,loader):
        num_samples = len(loader)
        num_batches_processed = 0
        for x,y in loader:
            x = resize(x,(224,224))
            if (x.shape[1] == 1):
                x = x.repeat(1,3,1,1)
            if torch.cuda.is_available():
                x = x.cuda()
            self.model(x)
            num_batches_processed += 1
            print(f'Processed {num_batches_processed}/{num_samples} batches', end='\r')
        features = torch.cat(self.features).cpu().numpy()
        self.features = []
        return features
    
    def fit(self,X,y):
        self.svm.fit(X,y.reshape(-1))
    
    def predict(self,X):
        self.model(X)
        return self.svm.predict(self.features[0].cpu().numpy().reshape(1,-1))
