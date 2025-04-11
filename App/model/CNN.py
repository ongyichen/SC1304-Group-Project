import torch.nn as nn

# Train from scratch Architecture 
class CNN(nn.Module):
    
    def __init__(self, inChannels, numClasses):

        super().__init__()
        
        self.conv1 = self.ConvBlock(inChannels, 64)
        self.conv2 = self.ConvBlock(64, 128, pool=True) 
        self.res1 = nn.Sequential(self.ConvBlock(128, 128), self.ConvBlock(128, 128))
        
        self.conv3 = self.ConvBlock(128, 256, pool=True) 
        self.conv4 = self.ConvBlock(256, 512, pool=True)
        
        self.res2 = nn.Sequential(self.ConvBlock(512, 512), self.ConvBlock(512, 512))
        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                       nn.Flatten(),
                                       nn.Linear(512, numClasses))
        
    # Convolution block with BatchNormalization
    def ConvBlock(self, inChannels, outChannels, pool=False):
        layers = [nn.Conv2d(inChannels, outChannels, kernel_size=3, padding=1),
                nn.BatchNorm2d(outChannels),
                nn.ReLU(inplace=True)]
        if pool:
            layers.append(nn.MaxPool2d(4))
        return nn.Sequential(*layers)
        
    def forward(self, x): # x is the loaded batch
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out  