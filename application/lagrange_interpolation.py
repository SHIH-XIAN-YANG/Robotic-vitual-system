import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import csv

sys.path.append('..')
sys.path.insert(1,"../rt605")
sys.path.insert(1,"../mismatch_classification")

from rt605 import RT605
from libs.ServoMotor import ServoMotor
from libs.type_define import*

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

def load_model_and_infer(model_path, input_data):
    # Assuming input_data is a torch tensor with appropriate shape
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Initialize the model
    input_shape = input_data.shape[1]  # assuming the last dimension is the input size
    output_shape = 6  # Set this to the correct output size
    model = CNN1D(input_shape, output_shape).to(device)
    print(input_data.shape)

    # Load the model weights
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode

    input_data = input_data.to(device)
    # Perform inference
    with torch.no_grad():  # No need to track gradients for inference
        output = model(input_data)
    
    return output

def find_closest_index(data, target):
    min_diff = float('inf')
    closest_index = -1

    quarter_len = len(data) // 4
    three_quarter_len = (len(data)*3) // 4

    for idx in range(quarter_len, three_quarter_len):
        value = data[idx]
        diff = abs(value - target)

        if diff<min_diff:
            min_diff = diff
            closest_index = idx
    return closest_index


def lagrange_Interpolation(x_data, y_data, min_x, max_x, pointcount=1000):
    opt_gain = 0.0
    min_y = np.inf

    for idx in range(pointcount):
        x = min_x + idx * (max_x - min_x)/pointcount
        y = 0.0

        # compute y value of polynomial(x)
        for i in range(len(x_data)):
            term = y_data[i]
            for j in range(len(x_data)):
                if j != i:
                    term = term * (x - x_data[j])/(x_data[i] - x_data[j])
            y = y + term
        if y < min_y:
            min_y = y
            opt_gain = x

    return opt_gain

def main():
    rt605 = RT605()

    path_file_dir = '../data/Path/'
    path_name = "sine_f6_full_joints.txt"

    rt605.load_HRSS_trajectory(path_file_dir + path_name)

    rt605.compute_GTorque.enable_Gtorq(en=True)
    rt605.compute_friction.enable_friction(en=True)


    rt605.run_HRSS_intp()


    gain_deviation = np.zeros(6)

    for idx in range(6):
        # print(max(rt605.q_c[:,idx]), max(rt605.q[:, idx]))
        gain_deviation[idx] = max(rt605.q_c[:,idx]) - max(rt605.q[:, idx])

    print(gain_deviation)

    phase_deviation = np.zeros(6)
    for idx in range(6):
        # print(find_closest_index(rt605.q[:, idx], rt605.q_c[0, idx]), find_closest_index(rt605.q_c[:, idx], rt605.q_c[0, idx]))
        phase_deviation[idx] = rt605.ts * (find_closest_index(rt605.q[:, idx], rt605.q_c[0, idx]) - find_closest_index(rt605.q_c[:, idx], rt605.q_c[0, idx]))
    print(phase_deviation)



    # Determine the lowest joint
    model_path = '6_12_17_23_best_model_acc_95.875.pth'
    input_data = rt605.q_pos_err # size: (datasize , 6)
    # Convert the NumPy array to a PyTorch tensor
    input_data = torch.from_numpy(input_data).float()
    # Reshape the tensor to the desired size (1, 6, 1000)
    input_data = input_data.permute(1, 0).unsqueeze(0)
    output = load_model_and_infer(model_path, input_data)
    lag_joint = torch.argmax(output).item()
    print(f"lag joint: {lag_joint}")

    # Lagrange interpolation
    gain_deviation = []
    gain_deviation.append(max(rt605.q_c[:,lag_joint]) - max(rt605.q[:, lag_joint]))


    # Tuen Kpp gain
    #define min max value of gain
    k_min = 5
    k_max = 500
    r_max = 1
    r_min = 0

    iters = 10
    kpp = []
    kpi = []
    kvp = []
    kvi = []

    kpp.append(rt605.joints[lag_joint].get_PID(ServoGain.Position.value.kp))
    kpi.append(rt605.joints[lag_joint].get_PID(ServoGain.Position.value.ki))
    kvp.append(rt605.joints[lag_joint].get_PID(ServoGain.Velocity.value.kp))
    kvi.append(rt605.joints[lag_joint].get_PID(ServoGain.Velocity.value.ki))

    kpp.append(kpp[0] + gain_deviation[0]/(r_max - r_min) * (k_max - k_min))
    rt605.setPID(lag_joint,"kpp", kpp[1])

    rt605.resetServoDrive()

    rt605.run_HRSS_intp()
    gain_deviation.append(max(rt605.q_c[:,lag_joint]) - max(rt605.q[:, lag_joint]))
    rt605.resetServoDrive()

    for iter in range(2, iters):
        
        kpp.append(lagrange_Interpolation(kpp, gain_deviation, k_min, k_max, 1000))
        rt605.run_HRSS_intp()
        gain_deviation.append(max(rt605.q_c[:,lag_joint]) - max(rt605.q[:, lag_joint]))
        rt605.resetServoDrive()
        print(f"kpp: {kpp[iter]} || gain_deviation: {gain_deviation[iter]}")

    
    save_file_path = 'gain.csv'

    # Write to CSV
    with open(save_file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write the header
        csvwriter.writerow(['x', 'y'])
        # Write the data
        for x, y in zip(kpp, gain_deviation):
            csvwriter.writerow([x, y])

    print(f"Data successfully written to {save_file_path}")



if __name__ == "__main__":
    main()