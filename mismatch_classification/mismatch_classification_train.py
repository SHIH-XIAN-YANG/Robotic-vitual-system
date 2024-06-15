#%%
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt

import json
import numpy as np
import pymysql
import time
#%% Construct Neural Network
class NeuralNetwork(nn.Module):
    def __init__(self, ipnut_shape, output_shape):
        super().__init__()
        # Input layer (assuming batch size 1)
        self.input_layer = nn.Flatten()  # If input data is not already flattened

        # Hidden layer (customize size and activation)
        self.fc1 = nn.Linear(ipnut_shape, 256) 
        self.act1 = nn.Sigmoid()  
        self.fc2 = nn.Linear(256, 128)
        self.act2 = nn.Sigmoid()
        self.fc3 = nn.Linear(128,64)
        

        # Output layer
        self.output_layer = nn.Linear(64, output_shape)  # Example with 6 output neurons

    def forward(self, x):
        # x = self.input_layer(x)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        x = torch.softmax(x, dim=-1)  
        x = self.output_layer(x)
        return x


#%%

# connect to databse
connction = pymysql.connect(
    host = "localhost",
    user="root",
    port=3306,
    password="Sam512011",
    database="bw_mismatch_db"
)

cursor = connction.cursor()


"""
+--------------------------------------------------------------------------------------------------------------------------------
|  id |  Gain | BW | C_error | min_bandwidth | t_errX | t_errY | t_errZ | tracking_err_pitch | tracking_err_roll | tracking_err_yaw | contour_err_img_path | ori_contour_err_img_path
|-----|-------|----|---------|---------------|--------|--------|--------|
...

"""

start_time = time.time()

sql = "SELECT Gain FROM bw_mismatch_data WHERE min_bandwidth <= 4"
cursor.execute(sql)
gain = np.array([])
data = cursor.fetchall()
for idx, row in enumerate(data):
    # print(json.loads(row[0]))
    gain = np.append(gain, json.loads(row[0]),axis=0)
gain = gain.reshape((-1,len(json.loads(row[0]))))


sql = "SELECT Bandwidth FROM bw_mismatch_data WHERE min_bandwidth <= 4"
cursor.execute(sql)
bandwidth = np.array([])
data = cursor.fetchall()
for idx, row in enumerate(data):
    # print(json.loads(row[0]))
    bandwidth = np.append(bandwidth, json.loads(row[0]))
bandwidth = bandwidth.reshape((-1,len(json.loads(row[0]))))

# print(bandwidth)
sql = "SELECT max_bandwidth FROM bw_mismatch_data WHERE min_bandwidth <= 4"
cursor.execute(sql)
min_bandwidth = np.array([])
data = cursor.fetchall()
for idx, row in enumerate(data):
    # print(row[0])
    arr = [0]*6
    arr[row[0]-1] = 1
    min_bandwidth = np.append(min_bandwidth,[arr])
min_bandwidth = min_bandwidth.reshape((-1, len(arr)))

sql = "SELECT contour_err FROM bw_mismatch_data WHERE min_bandwidth <= 4"
cursor.execute(sql)
contour_err = np.array([])
data = cursor.fetchall()
for idx, row in enumerate(data):
    # print(len(json.loads(row[0])))
    contour_err = np.append(contour_err, json.loads(row[0]))
# print(row)
contour_err = contour_err.reshape((-1, len(json.loads(row[0]))))

sql = "SELECT ori_contour_err FROM bw_mismatch_data WHERE min_bandwidth <= 4"
cursor.execute(sql)
ori_contour_err = np.array([])
data = cursor.fetchall()
for idx, row in enumerate(data):
    # print(len(json.loads(row[0])))
    ori_contour_err = np.append(ori_contour_err, json.loads(row[0]))
# print(row)
ori_contour_err = ori_contour_err.reshape((-1, len(json.loads(row[0]))))


end_time = time.time()
execution_time = end_time - start_time

print(f"Execution time: {execution_time:.2f} seconds")


#%%

"""
input data: contour_err
output(predict result): max_bandwidth

"""

contour_err = np.stack((contour_err, ori_contour_err),axis=1)
print((contour_err.shape[1]))
print((min_bandwidth.shape))
print((gain.shape))
print((bandwidth.shape))

#%%

epochs = 10
batch_size = 32



input_shape = (contour_err.shape[2]) #length of row in array contouring_err
output_shape = (min_bandwidth.shape[1])

# Create dataset using a custom class (optional, but recommended for flexibility)
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]
    
# Split data into train and test sets
train_size = int(0.8 * contour_err.shape[0])
test_size = contour_err.shape[0] - train_size

print(train_size, test_size)
# Create the combined dataset
dataset = CustomDataset(contour_err, min_bandwidth)

# Perform train-test split using random_split for better shuffling
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Configure dataloaders for training and testing
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # Adjust batch size as needed
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # No shuffling for testing


# train_data, test_data = contour_err[:train_size], contour_err[train_size:]
# train_target, test_target = max_bandwidth[:train_size], max_bandwidth[train_size:]

# # Create TensorDatasets and Dataloaders
# train_dataset = TensorDataset(train_data, train_target)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# test_dataset = TensorDataset(test_data, test_target)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



# Define model and optimizer
# Define device (use GPU if available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# Define loss function
criterion = nn.CrossEntropyLoss()


# Create the network and optimizer
model = NeuralNetwork(input_shape, output_shape)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters())


# Train the model
for epoch in range(epochs):
    for data, target in train_dataloader:
        if torch.is_tensor(data) and data.dtype == torch.double:
            data = data.float()
            target  = target.long()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        output = model(data)
        loss = criterion(output, target.squeeze())
        loss.backward()
        optimizer.step()

    # Evaluate on test set (optional)
    with torch.no_grad():
        correct = 0
        total = 0
        for data, target in test_dataloader:
            
            if torch.is_tensor(data) and data.dtype == torch.double:
                data = data.float()
                target = target.long()
            data, target = data.to(device), target.to(device)
            output = model(data)
            print(output.data.shape)
            total += target.size(0)
            predicted = torch.argmax(output.data, dim=1)
            target = torch.argmax(target, dim=1)
            # print(f'predice t= {predicted}')
            
            # print(predicted)
            # print(target)
            correct += (predicted == target).sum().item()

        print(f"Epoch {epoch+1} accuracy: {correct}/{total:.4f}")

# Save the model (optional)
torch.save(model.state_dict(), "model.pth")

# Save the model
# model.save(f'saved_model/NN/arch_{first_layer_node_number}_{second_layer_node_number}_train_acc_{training_acc[-1]:.3f}_val_acc_{val_acc[-1]:.3f}.h5')
# %%

cursor.close()
connction.close()


