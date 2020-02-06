import torch

# Define CNN
import torch.nn as nn
import torch.nn.functional as F

class BaselineCNN(nn.Module):
    def __init__(self, input_shape=(48, 48), verbose=False):
        super(BaselineCNN, self).__init__()
        nb_feature = 8
        self.conv1 = nn.Conv2d(1, nb_feature, 3, padding=1)
        self.conv2 = nn.Conv2d(nb_feature, nb_feature, 3, padding=1)
        self.conv3 = nn.Conv2d(nb_feature, 2*nb_feature, 3, padding=1)
        self.conv4 = nn.Conv2d(2*nb_feature, 2*nb_feature, 3, padding=1)
        self.conv5 = nn.Conv2d(2*nb_feature, 4*nb_feature, 3, padding=1)
        self.conv6 = nn.Conv2d(4*nb_feature, 4*nb_feature, 3, padding=1)
        self.conv7 = nn.Conv2d(4*nb_feature, 8*nb_feature, 3, padding=1)
        self.conv8 = nn.Conv2d(8*nb_feature, 8*nb_feature, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(576, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 7)
        
        if verbose:
            dummy_input = torch.randn(input_shape).unsqueeze(0).unsqueeze(0)
            f = self.get_features(dummy_input)
            x = f.view(-1, 576)
            out = self.get_classifier(x)
            print(f"Shape after the conv layers is: {f.shape}")
            print(f"Final output for a random input is: {out} of shape: {out.shape}")
        
    
    def get_features(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout(self.pool(x))

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.dropout(self.pool(x))

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.dropout(self.pool(x))

        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.dropout(self.pool(x))
        return x
    
    def get_classifier(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


    def forward(self, x):
        x = self.get_features(x)
        
        # print(x.shape)
        x = x.view(-1, 576)
        
        x = self.get_classifier(x)
        
        return x
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            x = self.forward(x)
        return nn.Softmax(dim=7)(x)
        

