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
#sys.path.insert(1,"../rt605")
sys.path.append("../mismatch_classification/")

from rt605 import RT605
from libs.ServoMotor import ServoMotor
from libs.type_define import*

from mismatch_classification.model_playground import *

from matplotlib import pyplot as plt


class Intp():
    model: CNN1D
    model_weight_path: str
    device: torch.device


    rt605:RT605
    trajectory_path: str
    q_pos_err: torch.Tensor
    lag_joint: int

    gain_deviation: list
    max_gain_deviation: float
    min_gain_deviation: float

    phase_shift: list
    max_phase_shift: float
    min_phase_shift: float

    iter:int
    gain_mode: ServoGain
    gain: list

    save_file: str

    def __init__(self, iter:int=10):
        self.rt605 = RT605()
        self.trajectory_path = "../data/Path/"+"sine_f6_full_joints.txt"

        self.rt605.load_HRSS_trajectory(self.trajectory_path)
        self.rt605.compute_GTorque(en=True)
        self.rt605.compute_friction(en=True)


        # Run model inference: get the initial lag joints values
        self.model_weight_path = '6_12_17_23_best_model_acc_95.875.pth'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.model = CNN1D(input_dim=6, output_dim=6).to(self.device)
        self.model.load_state_dict(torch.load(self.model_weight_path))
        self.model.eval()

        self.max_gain_deviation = 0.1
        self.min_gain_deviation = -0.1
        self.max_phase_shift = 0.1
        self.min_phase_shift = -0.1

        self.iter = iter
        
    
    def inference(self)->None:
        self.rt605.run_HRSS_intp()
        self.q_pos_err = torch.from_numpy(self.rt605.q_pos_err).float().permute(1,0).unsqueeze(0)
        self.q_pos_err.to(self.device)

        with torch.no_grad():
            output = self.model(self.q_pos_err)
        
        self.lag_joint = torch.argmax(output).item()
        print(f"Initial lag joint: {self.lag_joint}")
    
    def lagrange_intp(self, x_data, y_data, min_x, max_x, pointcount=1000)->float:
        opt_gain = 0.0
        min_y = np.inf

        for idx in range(pointcount):
            x = min_x + idx * (max_x - min_x)/pointcount
            y = 0.0

            # compute y value of polynomial(x)
            for i in range(len(x_data)):
                term = y_data[i]
                for j in range(len(x_data)):
                    if (j != i) and (x_data[i] != x_data[j]):
                        term = term * (x - x_data[j])/(x_data[i] - x_data[j])
                    
                y = y + term
            if abs(y) < min_y:
                min_y = abs(y)
                opt_gain = x

        return opt_gain
    
    def run(self, tune_mode: ServoGain, k_min:float, k_max:float):
        self.gain_mode = tune_mode
        
        
        self.gain_deviation.append(max(self.rt605.q_c[:, self.lag_joint]) - max(self.rt605.q[:, self.lag_joint]))

        self.gain.append(self.rt605.joints[self.lag_joint].get_PID(tune_mode))

        gain_init = self.gain[0] + self.gain_deviation[0]/(k_max - k_min) * (self.max_gain_deviation - self.min_gain_deviation)
        self.gain.append(gain_init)

        self.rt605.setPID(self.lag_joint, tune_mode, gain_init)
        self.rt605.run_HRSS_intp()

        self.gain_deviation.append(max(self.rt605.q_c[:, self.lag_joint]) - max(self.rt605.q[:, self.lag_joint]))

        for iter in range(2, self.iter):
            next_gain = self.lagrange_intp(self.gain, self.gain_deviation, k_min, k_max, 10000)
            self.gain.append(next_gain)
            self.rt605.setPID(self.lag_joint, ServoGain.Position.value.kp, next_gain)
            self.rt605.run_HRSS_intp()
            # rt605.plot_joint(True)
            self.gain_deviation.append(max(self.rt605.q_c[:,self.lag_joint]) - max(self.rt605.q[:, self.lag_joint]))
            # plt.figure()
            # plt.plot(rt605.time, rt605.q_c[:,lag_joint], label="ref")
            # plt.plot(rt605.time, rt605.q[:, lag_joint], label="act")
            # plt.legend()
            # plt.show()
            print(f"kpp: {self.gain[iter]} || gain_deviation: {self.gain_deviation[iter]}")

    def save_file(self, save_file_path:str=None):
        if save_file_path is not None:
            # Write to CSV
            with open(save_file_path, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                # Write the header
                csvwriter.writerow(['x', 'y'])
                # Write the data
                for x, y in zip(self.gain, self.gain_deviation):
                    csvwriter.writerow([x, y])

            print(f"Data successfully written to {save_file_path}")

def main():
    interpolation = Intp()
    
    for iter in range(0, 6):
    
        interpolation.inference()
        
        interpolation.run(ServoGain.Velocity.value.kp, 0, 100)

        interpolation.run(ServoGain.Velocity.value.ki, 0, 100)

        interpolation.run(ServoGain.Position.value.kp, 0, 100)

        interpolation.run(ServoGain.Position.value.ki, 0, 100)



if __name__ == "__main__":
    main()