# This file contains the models used for both parts of the assignment:
#
#   - DCGenerator       --> Used in the vanilla GAN in Part 1
#   - CycleGenerator    --> Used in the CycleGAN in Part 2
#   - DCDiscriminator   --> Used in both the vanilla GAN and CycleGAN (Parts 1 and 2)
#
# For the assignment, you are asked to create the architectures of these three networks by
# filling in the __init__ methods in the DCGenerator, CycleGenerator, and DCDiscriminator classes.
# Note that the forward passes of these models are provided for you, so the only part you need to
# fill in is __init__.

import pdb
import torch        
import torch.nn as nn
import torch.nn.functional as F


def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a transposed-convolutional layer, with optional batch normalization.
    """
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True, init_zero_weights=False):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    if init_zero_weights:
        conv_layer.weight.data = torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.001
    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


class DCGenerator(nn.Module):
    def __init__(self, noise_size, conv_dim):
        super(DCGenerator, self).__init__()

        ###########################################
        ##   FILL THIS IN: CREATE ARCHITECTURE   ##
        ###########################################
        ############## TODO #######################
        self.fc = nn.Linear(noise_size*1*1,128*4*4)
        self.deconv1, self.BatchNorm1 = deconv(128,64,kernel_size=4)
        self.ReLU1  = nn.ReLU()
        self.deconv2, self.BatchNorm2 = deconv(64,32,kernel_size=4)
        self.ReLU2 = nn.ReLU()
        self.deconv3,_ = deconv(32,3,kernel_size=4)
        self.tanh = nn.Tanh()


    def forward(self, z):
        """Generates an image given a sample of random noise.

            Input
            -----
                z: BS x noise_size x 1 x 1   -->  16x100x1x1            

            Output
            ------
                out: BS x channels x image_width x image_height  -->  16x3x32x32
        """
        ##############################################
        ################TODO##########################
         ################# Complete this forward method using deconv block code provided above ######
        bsize = z.size(0)
        z = z.reshape(z.size()[0],-1)
        z = self.fc(z)
        z = z.reshape(shape=[bsize,128,4,4])
        z = self.ReLU1(self.BatchNorm1(self.deconv1(z)))
        z = self.ReLU2(self.BatchNorm2(self.deconv2(z)))
        z = self.tanh(self.deconv3(z))
        #print("In DC Generator Forward function: ",z.size())
        return z


class ResnetBlock(nn.Module):
    def __init__(self, conv_dim):
        super(ResnetBlock, self).__init__()
        self.conv_layer,self.BatchNormR = conv(in_channels=conv_dim, out_channels=conv_dim, kernel_size=3, stride=1, padding=1)
        self.ReLUR = nn.ReLU()
    def forward(self, x):

        ############## TODO##############
        ###### Complete forward method###
        #################################
        return self.ReLUR(self.BatchNormR(self.conv_layer(x)))


class CycleGenerator(nn.Module):
    """Defines the architecture of the generator network.
       Note: Both generators G_XtoY and G_YtoX have the same architecture in this assignment.
    """
    def __init__(self, conv_dim=64, init_zero_weights=False):
        super(CycleGenerator, self).__init__()

        ###########################################
        ##   FILL THIS IN: CREATE ARCHITECTURE   ##
        ###########################################

        # Define the encoder part of the generator (that extracts features from the input image)
        self.conv1, self.BatchNorm1 = conv(3,32,kernel_size=4)
        self.ReLU1 = nn.ReLU()
        self.conv2, self.BatchNorm2 = conv(32,64,kernel_size=4)
        self.ReLU2 = nn.ReLU()
        self.deconv1, self.BatchNorm3 = deconv(64,32,kernel_size=4)
        self.ReLU3 = nn.ReLU()
        self.deconv2,_ = deconv(32,3,kernel_size=4)
        self.tanh = nn.Tanh()

    def forward(self, x):
        """Generates an image conditioned on an input image.

            Input
            -----
                x: BS x 3 x 32 x 32

            Output
            ------
                out: BS x 3 x 32 x 32
        """
        ################# Complete this forward method using conv, resnet and deconv blocks provided above ######
        x = self.ReLU1(self.BatchNorm1(self.conv1(x)))
        x = self.ReLU2(self.BatchNorm2(self.conv2(x)))
        res = ResnetBlock(conv_dim=64)
        res.cuda()
        x = res(x)
        x = self.ReLU3(self.BatchNorm3(self.deconv1(x)))
        x = self.tanh(self.deconv2(x))
        return x


class DCDiscriminator(nn.Module):
    """Defines the architecture of the discriminator network.
       Note: Both discriminators D_X and D_Y have the same architecture in this assignment.
    """
    def __init__(self, conv_dim=64):
        super(DCDiscriminator, self).__init__()
        ###########################################
        ##   FILL THIS IN: CREATE ARCHITECTURE   ##
        ###########################################

        self.Conv1,  self.BatchNorm1 = conv(3, 32, kernel_size=4)
        self.ReLU1 = nn.ReLU()
        self.Conv2 , self.BatchNorm2 = conv(32, 32*2, kernel_size=4)
        self.ReLU2 = nn.ReLU()
        self.Conv3,  self.BatchNorm3 = conv(32*2, 32*4, kernel_size=4)
        self.ReLU3 = nn.ReLU()
        self.fc = nn.Linear(128*4*4,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        ################# Complete this forward method using conv block code provided above ######
        x = self.ReLU1(self.BatchNorm1(self.Conv1(x)))
        x = self.ReLU2(self.BatchNorm2(self.Conv2(x)))
        x = self.ReLU3(self.BatchNorm3(self.Conv3(x)))
        x = x.view(x.size()[0],-1)
        x = self.sigmoid(self.fc(x))
        return x
