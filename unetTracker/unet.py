import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    """
    This class represent a group of 2 convolutions steps with Batch normalization

    The width and height of the output is 4 pixels less than the input.
    """
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
    Unet model
    
    in_channels: n channels in input image, usually the number of color channels
    out_channels: number of objects you want to segment
    
    """
    def __init__(self, in_channels = 3, out_channels = 1, features=[64,128,256,512]): 
        super(Unet, self).__init__()
        
        # We will use nn.ModuleList to hold our down and up processes
        self.downs = nn.ModuleList() # Does from input of image to the bottleneck
        self.ups = nn.ModuleList() # Does from bottleneck to the output
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2) # Reduce the image size by 2
        
        
        # Create the module list for the Down process
        # This is a series of double convolutions
        for feature in features:
            self.downs.append(DoubleConv(in_channels=in_channels,out_channels=feature))
            in_channels = feature
        
        
        # Create the module list for the Up process
        # We increase the image size with the ConvTranspose2d, then apply 2 convolutions
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(in_channels=feature*2,out_channels=feature,kernel_size=2, stride=2) # will double size of image
            )
            self.ups.append(
                DoubleConv(in_channels=feature*2,out_channels=feature) # *2 because of the concatenation 
            )
        
        # Bottom part of the U (bottleneck)
        self.bottleneck = DoubleConv(features[-1],features[-1]*2)

        # Last step given the right number of output segmentation masks
        self.finalConv = nn.Conv2d(in_channels=features[0],out_channels=out_channels,kernel_size=1,stride=1) # just change the number of channel, no real conv
        
    def forward(self,x):
        
        skip_connections = [] # We keep a list of the pocessed results of the downward path of the model. We will feed them to the upward steps

        # Loop for the steps from Input to bottleneck
        for down in self.downs: 
            x = down(x) # this is a double conv layer
            skip_connections.append(x) # store the output for the upward section of the model
            x = self.pool(x)
            
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1] # reverse the list (now from small to large image size)

        # Loop for the steps from the bottleneck to the Output
        for idx in range(0, len(self.ups), 2): # we do up 
            x = self.ups[idx](x) # apply doubleConv, increase the size of image by 2
            skip_connection = skip_connections[idx//2] 
            
            if x.shape != skip_connection.shape:
                # This happens if the image width and height size is not divisible by 16
                # This could affect performance but only by a few pixels.
                # print("resize()", x.shape,skip_connection.shape)
                x = TF.resize(x,size=skip_connection.shape[2:]) # we don't touch batch size and number of channels
                                                                    
            concat_skip = torch.cat((skip_connection,x),dim=1) # concatenate x and skip
            x = self.ups[idx+1](concat_skip) #second doubleConv
        
        return self.finalConv(x)
        