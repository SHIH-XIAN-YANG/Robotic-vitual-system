import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader



class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        images = [image[idx] for image in self.images]
        label = self.labels[idx]

        if self.transform:
            images = [self.transform(image) for image in images]

        concatenated_image = torch.cat(images, dim=0)
        return concatenated_image, label
    
class MobileNetV2(nn.Module):
    def __init__(self, in_channels=6, num_classes=6):
        super(MobileNetV2, self).__init__()
        # Initial convolution layer
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        # Inverted residual blocks
        self.block1 = self._make_block(32, 16, stride=1)
        self.block2 = self._make_block(16, 24, stride=2)
        self.block3 = self._make_block(24, 48, stride=2)
        self.block4 = self._make_block(48, 64, stride=2)
        self.block5 = self._make_block(64, 96, stride=1)
        self.block6 = self._make_block(96, 160, stride=2)
        self.block7 = self._make_block(160, 320, stride=2)

        # Final classification layer
        self.conv2 = nn.Conv2d(320, num_classes, kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_block(self, in_filters, out_filters, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_filters),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.avgpool(x)
        x = self.conv2(x)
        x = x.squeeze()
        return x
    

# Define the model architecture
class CustomResNet(nn.Module):
    def __init__(self, pretrained=True,input_channels=6):
        super(CustomResNet, self).__init__()
        self.base_model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        self.base_model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Replace first conv with 3*2 channels
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_ftrs, 6)

    def forward(self, x):
        x = self.base_model(x)
        return x
    
class UltraLightCNN(nn.Module):
    def __init__(self, input_shape=6, num_classes=6):
        super(UltraLightCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class CNN1D(nn.Module):
    def __init__(self, input_dim=5, output_dim=6):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv7 = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(16)
        self.batch_norm2 = nn.BatchNorm1d(32)
        self.batch_norm3 = nn.BatchNorm1d(64)
        self.batch_norm4 = nn.BatchNorm1d(128)
        self.batch_norm5 = nn.BatchNorm1d(256)
        self.batch_norm6 = nn.BatchNorm1d(512)
        self.batch_norm7 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.2)
        self.dropout4 = nn.Dropout(p=0.2)
        self.dropout5 = nn.Dropout(p=0.1)
        self.dropout6 = nn.Dropout(p=0.1)
        self.dropout7 = nn.Dropout(p=0.1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(83968, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(84544, output_dim)

    def forward(self, x):
        x = self.pool(self.relu(self.batch_norm1(self.conv1(x))))
        x = self.dropout1(x)
        x = self.pool(self.relu(self.batch_norm2(self.conv2(x))))
        x = self.dropout2(x)
        x = self.pool(self.relu(self.batch_norm3(self.conv3(x))))
        x = self.dropout3(x)
        # x = self.pool(self.relu(self.batch_norm4(self.conv4(x))))
        # x = self.dropout4(x)
        # x = self.pool(self.relu(self.batch_norm5(self.conv5(x))))
        # x = self.dropout5(x)
        # x = self.pool(self.relu(self.batch_norm6(self.conv6(x))))
        # x = self.dropout6(x)
        # x = self.pool(self.relu(self.batch_norm7(self.conv7(x))))
        # x = self.dropout7(x)
        x = self.flatten(x)
        # x = self.fc1(x)
        # x = self.fc2(x)
        x = self.fc3(x)
        return x
class multi_channel_CNN_v2(nn.Module):
    def __init__(self, num_classes=6, pipe_num=3, in_channels=3):
        super(multi_channel_CNN_v2, self).__init__()

        self.pipe_num = pipe_num
        self.num_classes = num_classes

        # Define convolutional layers for each input channel
        self.conv_layers = nn.ModuleList()
        for i in range(pipe_num):
            conv = nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.conv_layers.append(conv)

        # Fully connected layers after concatenation
        self.fc1 = nn.Linear(pipe_num * 32 * 150 * 150, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Iterate through each input image channel
        features = []
        for i in range(self.pipe_num):
            # Apply convolutional layers
            conv_out = self.conv_layers[i](x[i])
            features.append(conv_out)

        # Concatenate the feature maps
        x = torch.cat(features, dim=1)
        x = x.view(-1, self.pipe_num * 32 * 150 * 150)  # Reshape for fully connected layer

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
class multi_channel_CNN(nn.Module):
    pipe_num:int = None
    num_classed:int = None

    def __init__(self, num_classes=6, pipe_num=3):
        super(multi_channel_CNN, self).__init__()
        
        # Convolutional layers for image 1
        self.conv1_img1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2_img1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # self.conv3_img1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Convolutional layers for image 2
        self.conv1_img2 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2_img2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # self.conv3_img2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Convolutional layers for image 3
        self.conv1_img3 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2_img3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # self.conv3_img3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Convolutional layers for image 4
        self.conv1_img4 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2_img4 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # self.conv3_img4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Fully connected layers after concatenation
        #600 * 600 input image after two maxpool-->150*150
        self.fc1 = nn.Linear(pipe_num*32*150*150, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.pipe_num = pipe_num
        self.num_classes = num_classes
        
    def forward(self, x):
        if self.pipe_num == 4:
            x1,x2,x3,x4 = x[0], x[1], x[2], x[3]
        else:
            x1,x2,x3 = x[0], x[1], x[2]
        # Image 1 processing
        x1 = F.relu(self.conv1_img1(x1))
        x1 = F.max_pool2d(x1, 2)
        x1 = F.relu(self.conv2_img1(x1))
        x1 = F.max_pool2d(x1, 2)
        # x1 = F.relu(self.conv3_img1(x1))
        # x1 = F.max_pool2d(x1, 2)
        
        # Image 2 processing
        x2 = F.relu(self.conv1_img2(x2))
        x2 = F.max_pool2d(x2, 2)
        x2 = F.relu(self.conv2_img2(x2))
        x2 = F.max_pool2d(x2, 2)
        # x2 = F.relu(self.conv3_img2(x2))
        # x2 = F.max_pool2d(x2, 2)
        
        # Image 3 processing
        x3 = F.relu(self.conv1_img3(x3))
        x3 = F.max_pool2d(x3, 2)
        x3 = F.relu(self.conv2_img3(x3))
        x3 = F.max_pool2d(x3, 2)
        # x3 = F.relu(self.conv3_img3(x3))
        # x3 = F.max_pool2d(x3, 2)


        # Image 4 processing
        if self.pipe_num == 4:
            x4 = F.relu(self.conv1_img3(x4))
            x4 = F.max_pool2d(x4, 2)
            x4 = F.relu(self.conv2_img3(x4))
            x4 = F.max_pool2d(x4, 2)
            # x4 = F.relu(self.conv3_img3(x4))
            # x4 = F.max_pool2d(x4, 2)
        
        # Concatenate the feature maps
        if self.pipe_num == 3:
            x = torch.cat((x1, x2, x3), dim=1)
        else:
            x = torch.cat((x1, x2, x3, x4), dim=1)
        # print(x.shape)
        x = x.view(-1, self.pipe_num*32*150*150)  # Reshape for fully connected layer
        
        # print(x.shape)
        x = F.relu(self.fc1(x))

        x = self.fc2(x)
        
        return x

if __name__=="__main__":
    
    # Example of parallel CNN
    model = multi_channel_CNN_v2(num_classes=6, pipe_num=4)

    image1 = torch.randn(1, 3, 600, 600)
    image2 = torch.randn(1, 3, 600, 600)
    image3 = torch.randn(1, 3, 600, 600)
    image4 = torch.randn(1, 3, 600, 600)

    concatenated_image = torch.cat([image1, image2, image3, image4], dim=0)
    print(type(concatenated_image))

    output = model((image1, image2, image3, image4),)
    print(output.shape)  

    # CNN-1D example
    model = CNN1D(input_dim=6, output_dim=6)

    input_tensor = torch.randn((1, 6, 10568))  # Batch size 1, input dimension 5, sequence length 10
    output = model(input_tensor)
    print(output.shape)  

    


