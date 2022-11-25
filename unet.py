import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3, padding=1,stride=1,bias=False),
            nn.BatchNorm2d(out_channels),# bn would cancel the bias
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3, padding=1,stride=1,bias=False),
            nn.BatchNorm2d(out_channels),# bn would cancel the bias
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class Unet(nn.Module):
    """
    in_channels: n channels in input image
    out_channels: number of object you want to segment
    
    if input image width and height size is not divisible by 16, we 
    """
    def __init__(self, in_channels = 3, out_channels = 1, features=[64,128,256,512]): 
        super(Unet, self).__init__()
        
        # We will use nn.ModuleList to hold our down and up processes
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        
        
        # Create the module list for the Down process
        for feature in features:
            self.downs.append(DoubleConv(in_channels=in_channels,out_channels=feature))
            in_channels = feature
        
        
        # Create the module list for the Up process
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(in_channels=feature*2,out_channels=feature,kernel_size=2, stride=2) # will double size of image
            )
            self.ups.append(
                DoubleConv(in_channels=feature*2,out_channels=feature) # *2 because of the concatenation 
            )
        
        # Bottom part of the U (bottleneck)
        self.bottleneck = DoubleConv(features[-1],features[-1]*2)
        
        self.finalConv = nn.Conv2d(in_channels=features[0],out_channels=out_channels,kernel_size=1,stride=1) # just change the number of channel, no real conv
        
    def forward(self,x):
        
        skip_connections = []
        
        for down in self.downs:
            x = down(x)
            skip_connections.append(x) # first high resolution
            x = self.pool(x)
            
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1] # reverse the list
        
        for idx in range(0, len(self.ups), 2): # we do up 
            x = self.ups[idx](x) # apply doubleConv, increase the size of image by 2
            skip_connection = skip_connections[idx//2] 
            
            if x.shape != skip_connection.shape:
                # This happens if the image width and height size is not divisible by 16
                # This could affect performance but only by a few pixels.
                print("resize()", x.shape,skip_connection.shape)
                x = TF.resize(x,size=skip_connection.shape[2:]) # we don't touch batch size and number of channels
                                                                    
            concat_skip = torch.cat((skip_connection,x),dim=1) # concatenate x and skip
            x = self.ups[idx+1](concat_skip) #second doubleConv
        
        return self.finalConv(x)
        