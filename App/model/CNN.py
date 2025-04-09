import torch.nn as nn

# Convolution block with BatchNormalization
def ConvBlock(inChannels, outChannels, pool=False):
    layers = [nn.Conv2d(inChannels, outChannels, kernel_size=3, padding=1),
             nn.BatchNorm2d(outChannels),
             nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

# Train from scratch Architecture 
class CNN(nn.Module):
    def __init__(self, inChannels, numClasses):
        super(CNN, self).__init__()
        
        self.conv1 = ConvBlock(inChannels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True) 
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        
        self.conv3 = ConvBlock(128, 256, pool=True) 
        self.conv4 = ConvBlock(256, 512, pool=True)
        
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                       nn.Flatten(),
                                       nn.Linear(512, numClasses))
        
    def forward(self, x): # x is the loaded batch
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out        