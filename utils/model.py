import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os
# ConvBlock definition
def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

# Model Architecture
class ResNet9(nn.Module):
    def __init__(self, in_channels, num_diseases):
        super(ResNet9, self).__init__()
        
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)  # out_dim: 128 x 64 x 64
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        
        self.conv3 = ConvBlock(128, 256, pool=True)  # out_dim: 256 x 16 x 16
        self.conv4 = ConvBlock(256, 512, pool=True)  # out_dim: 512 x 4 x 4
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        
        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Linear(512, num_diseases)
        )
        
    def forward(self, xb):  # xb is the input batch
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

# Load the pickle file
model_path = r'C:\Users\SAMMY\Documents\PROJECT-PHASE-5\app\models\crop_recommendation_rf_model_saved.pkl'

# Correctly load the model in read-binary mode
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Save the loaded object back to a new file (not needed if there's no modification, but provided here as an example)
save_path = r'C:\Users\SAMMY\Documents\PROJECT-PHASE-5\app\models\crop_recommendation_rf_model_saved.pkl'
with open(save_path, 'wb') as file:
    pickle.dump(model, file)
import os

model_path = r'C:\Users\SAMMY\Documents\PROJECT-PHASE-5\app\models\crop_recommendation_rf_model_saved.pkl'

if not os.path.exists(model_path):
    print("Error: The file does not exist at the specified path:", model_path)
else:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
