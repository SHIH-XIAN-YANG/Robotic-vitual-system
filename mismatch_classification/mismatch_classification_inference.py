import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchsummary import summary
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import json
import numpy as np
import time
from tqdm import tqdm

class CNN1D(nn.Module):
    def __init__(self, input_dim=6, output_dim=6):
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
        self.fc3 = nn.Linear(23040, output_dim)

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


# Define the function to load model weights and perform inference
def load_model_and_infer(model_path, input_data):
    # Assuming input_data is a torch tensor with appropriate shape
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # Initialize the model
    input_shape = input_data.shape[-1]  # assuming the last dimension is the input size
    output_shape = 6  # Set this to the correct output size
    model = CNN1D(input_shape, output_shape).to(device)


    # Load the model weights
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode

    input_data = input_data.to(device)
    # Perform inference
    with torch.no_grad():  # No need to track gradients for inference
        output = model(input_data)
    
    return output


model_path = 'model_weights.pth'
input_data = torch.randn(6, 1280)  # Example input data with batch size 1 and input size 128

output = load_model_and_infer(model_path, input_data)
print(output)