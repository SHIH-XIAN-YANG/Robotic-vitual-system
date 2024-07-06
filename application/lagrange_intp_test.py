import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import csv
import json
import json

sys.path.append('..')
#sys.path.insert(1,"../rt605")
sys.path.append("../mismatch_classification/")

from rt605 import RT605
from libs.ServoMotor import ServoMotor
from libs.type_define import*

from mismatch_classification.model_playground import *

from matplotlib import pyplot as plt


class FeatureType(Enum):
    magnitude_deviation = 0
    phase_shift = 1

class Loop_Type(Enum):
    pos = 0
    vel = 1

class Intp():
    model: CNN1D
    model_weight_path: str
    device: torch.device


    rt605:RT605
    trajectory_path: str
    q_pos_err: torch.Tensor
    lag_joint: int

    feature_type: FeatureType
    
    magnitude_deviation: list
    max_magnitude_deviation: float
    min_magnitude_deviation: float
    feature_type: FeatureType
    
    magnitude_deviation: list
    max_magnitude_deviation: float
    min_magnitude_deviation: float

    phase_shift: list
    max_phase_shift: float
    min_phase_shift: float



    feature: list
    max_feature: float
    min_feature: float

    max_gain: float
    min_gain: float



    feature: list
    max_feature: float
    min_feature: float

    max_gain: float
    min_gain: float

    iter:int
    tune_mode: ServoGain
    tune_mode: ServoGain
    gain: list

    save_file: str

    tuned_history: dict

    tune_loop_type: Loop_Type

    def __init__(self, iter:int=10):
        self.rt605 = RT605()
        self.trajectory_path = "../data/Path/"+"sine_f6_full_joints.txt"

        self.rt605.load_HRSS_trajectory(self.trajectory_path)
        self.rt605.compute_GTorque.enable_Gtorq(en=True)
        self.rt605.compute_friction.enable_friction(en=True)


        # Run model inference: get the initial lag joints values
        # Initial mismatch classification model
        self.model_weight_path = '6_12_17_23_best_model_acc_95.875.pth'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.model = CNN1D(input_dim=6, output_dim=6).to(self.device)
        self.model.load_state_dict(torch.load(self.model_weight_path))
        self.model.eval()

        self.max_magnitude_deviation = 10
        self.min_magnitude_deviation = -5
        self.max_phase_shift = 5
        self.min_phase_shift = -0.1

        self.iter = iter

        self.magnitude_deviation = []
        self.magnitude_deviation = []
        self.phase_shift = []
        self.gain = []

        self.features = []
        self.max_feature = 2
        self.min_feature = -2
        
        self.max_gain = 0.0
        self.min_gain = 0.0

        self.feature_type = None

        self.tuned_history = {}
        self.tuned_history["Kpp"] = {}
        self.tuned_history["Kpi"] = {}
        self.tuned_history["Kvp"] = {}
        self.tuned_history["Kvi"] = {}

        self.features = []
        self.max_feature = 2
        self.min_feature = -2
        
        self.max_gain = 0.0
        self.min_gain = 0.0

        self.feature_type = None

        self.tuned_history = {}
        self.tuned_history["Kpp"] = {}
        self.tuned_history["Kpi"] = {}
        self.tuned_history["Kvp"] = {}
        self.tuned_history["Kvi"] = {}

        tune_loop_type = Loop_Type.vel
    
    def inference(self)->None:
        self.rt605.run_HRSS_intp()
        self.q_pos_err = torch.from_numpy(self.rt605.q_pos_err).float()
        self.q_pos_err = self.q_pos_err.permute(1, 0).unsqueeze(0)
        self.q_pos_err = self.q_pos_err.to(self.device)

        with torch.no_grad():
            output = self.model(self.q_pos_err)
        
        self.lag_joint = torch.argmax(output).item()
        print(f"Initial lag joint: {self.lag_joint}")

    
    
    def lagrange_2D(self, x_data, y_data, z_data, min_x, max_x, min_y, max_y, pointcount=1000)->tuple:
        result = 0
        min_z = np.inf
        def lagrange_basis_polynomial(x, x_points, i):
            basis = 1
            for j in range(len(x_points)):
                if i != j:
                    basis *= (x - x_points[j]) / (x_points[i] - x_points[j])
            return basis
        
        opt_x, opt_y = 0.0,0.0

        for idx_x in range(pointcount):
            x = min_x + idx_x * (max_x - min_x)/pointcount

            for idx_y in range(pointcount):
                y = min_y + idx_y * (max_y - min_y)/pointcount

                for i in range(len(x_data)):
                    for j in range(len(y_data)):
                        L_i = lagrange_basis_polynomial(x, x_data, i)
                        M_j = lagrange_basis_polynomial(y, y_data, j)
                        result += z_data[i][j] * L_i * M_j
                if result < min_z:
                    min_z = result
                    opt_x = x
                    opt_y = y

        return opt_x, opt_y

    def lagrange_intp(self, x_data, y_data, min_x, max_x, pointcount=10000)->float:
        opt_gain = 0.0
        min_y = np.inf
        self.max_gain = max_x
        self.min_gain = min_x
        self.max_gain = max_x
        self.min_gain = min_x

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
    
    def compute_magnitude_deviation(self):
        return max(self.rt605.q_c[:, self.lag_joint]) - max(self.rt605.q[:, self.lag_joint])

    def compute_phase_shift(self):
        def find_closest_index(data, target):
            """
            find the closest index of trajectory's zeros crossing point
            
            """
            min_diff = float('inf')
            closest_index = -1

            quarter_len = len(data) // 4
            three_quarter_len = (len(data)*3) // 4

            for idx in range(quarter_len, len(data)):
                value = data[idx]
                diff = abs(value - target)

                if diff<min_diff:
                    min_diff = diff
                    closest_index = idx
            return closest_index
        phase_shift = self.rt605.ts * (find_closest_index(self.rt605.q[:, self.lag_joint], self.rt605.q_c[0, self.lag_joint]) \
                                        - find_closest_index(self.rt605.q_c[:, self.lag_joint], self.rt605.q_c[0, self.lag_joint]))

        return phase_shift
    
    def run(self, tune_mode: ServoGain, k_min:float, k_max:float, feature_type: FeatureType=FeatureType.magnitude_deviation):
        self.tune_mode = tune_mode
        self.max_gain = k_max
        self.min_gain = k_min
        self.feature_type = feature_type
        
        self.magnitude_deviation.append(self.compute_magnitude_deviation())
        self.phase_shift.append(self.compute_phase_shift())
    def compute_magnitude_deviation(self):
        return max(self.rt605.q_c[:, self.lag_joint]) - max(self.rt605.q[:, self.lag_joint])

    def compute_phase_shift(self):
        def find_closest_index(data, target):
            """
            find the closest index of trajectory's zeros crossing point
            
            """
            min_diff = float('inf')
            closest_index = -1

            quarter_len = len(data) // 4
            three_quarter_len = (len(data)*3) // 4

            for idx in range(quarter_len, len(data)):
                value = data[idx]
                diff = abs(value - target)

                if diff<min_diff:
                    min_diff = diff
                    closest_index = idx
            return closest_index
        phase_shift = self.rt605.ts * (find_closest_index(self.rt605.q[:, self.lag_joint], self.rt605.q_c[0, self.lag_joint]) \
                                        - find_closest_index(self.rt605.q_c[:, self.lag_joint], self.rt605.q_c[0, self.lag_joint]))

        return phase_shift
    
    def tune_gain(self, tune_mode: ServoGain, k_min:float, k_max:float, feature_type: FeatureType=FeatureType.magnitude_deviation):
        self.tune_mode = tune_mode
        self.max_gain = k_max
        self.min_gain = k_min
        self.feature_type = feature_type
        
        self.magnitude_deviation.append(self.compute_magnitude_deviation())
        self.phase_shift.append(self.compute_phase_shift())

        self.gain.append(self.rt605.joints[self.lag_joint].get_PID(tune_mode))

        self.print_gain(iter=0)

        # Compute initial gain
        if self.feature_type == FeatureType.magnitude_deviation:
            gain_init = k_min + self.magnitude_deviation[0]*(k_max - k_min) / (self.max_magnitude_deviation - self.min_magnitude_deviation)
        else:
            gain_init = k_min + self.phase_shift[0]*(k_max - k_min) / (self.max_phase_shift - self.min_phase_shift)

        if gain_init < k_min:
            gain_init = k_min
        elif gain_init > k_max:
            gain_init = k_max

        self.gain.append(gain_init)
        self.rt605.setPID(self.lag_joint, tune_mode, gain_init)
        self.rt605.run_HRSS_intp()

        if self.feature_type == FeatureType.magnitude_deviation:
            self.magnitude_deviation.append(self.compute_magnitude_deviation())
        else:
            self.phase_shift.append(self.compute_phase_shift())

        self.print_gain(iter=1)

        if self.feature_type == FeatureType.magnitude_deviation:
            self.magnitude_deviation.append(self.compute_magnitude_deviation())
        else:
            self.phase_shift.append(self.compute_phase_shift())

        self.print_gain(iter=1)

        # Start lagrange interpolation
        for iter in range(2, self.iter):
            if self.feature_type == FeatureType.magnitude_deviation:
                next_gain = self.lagrange_intp(self.gain, self.magnitude_deviation, k_min, k_max, 10000)
                self.gain.append(next_gain)
            
                self.rt605.setPID(self.lag_joint, tune_mode, next_gain)
                self.rt605.run_HRSS_intp()
            
                self.magnitude_deviation.append(self.compute_magnitude_deviation())
            else:
                next_gain = self.lagrange_intp(self.gain, self.phase_shift, k_min, k_max, 10000)
                self.gain.append(next_gain)
            
                self.rt605.setPID(self.lag_joint, tune_mode, next_gain)
                self.rt605.run_HRSS_intp()
            
                self.phase_shift.append(self.compute_phase_shift())
            self.print_gain(iter)
            # self.rt605.plot_joint(True)
            if self.feature_type == FeatureType.magnitude_deviation:
                next_gain = self.lagrange_intp(self.gain, self.magnitude_deviation, k_min, k_max, 10000)
                self.gain.append(next_gain)
            
                self.rt605.setPID(self.lag_joint, tune_mode, next_gain)
                self.rt605.run_HRSS_intp()
            
                self.magnitude_deviation.append(self.compute_magnitude_deviation())
            else:
                next_gain = self.lagrange_intp(self.gain, self.phase_shift, k_min, k_max, 10000)
                self.gain.append(next_gain)
            
                self.rt605.setPID(self.lag_joint, tune_mode, next_gain)
                self.rt605.run_HRSS_intp()
            
                self.phase_shift.append(self.compute_phase_shift())
            self.print_gain(iter)
            # self.rt605.plot_joint(True)
            # plt.figure()
            # plt.plot(rt605.time, rt605.q_c[:,lag_joint], label="ref")
            # plt.plot(rt605.time, rt605.q[:, lag_joint], label="act")
            # plt.legend()
            # plt.show()
        
        if self.tune_mode == ServoGain.Position.value.kp:
            self.tuned_history["Kpp"]["gain"] = self.gain
            if self.feature_type == FeatureType.magnitude_deviation:
                self.tuned_history["Kpp"]["magnitude deviation"] = self.magnitude_deviation
            elif self.feature_type == FeatureType.phase_shift:
                self.tuned_history["Kpp"]["phase shift"] = self.phase_shift
        elif self.tune_mode == ServoGain.Position.value.ki:
            self.tuned_history["Kpi"]["gain"] = self.gain
            if self.feature_type == FeatureType.magnitude_deviation:
                self.tuned_history["Kpi"]["magnitude deviation"] = self.magnitude_deviation
            elif self.feature_type == FeatureType.phase_shift:
                self.tuned_history["Kpi"]["phase shift"] = self.phase_shift
        elif self.tune_mode == ServoGain.Velocity.value.kp:
            self.tuned_history["Kvp"]["gain"] = self.gain
            if self.feature_type == FeatureType.magnitude_deviation:
                self.tuned_history["Kvp"]["magnitude deviation"] = self.magnitude_deviation
            elif self.feature_type == FeatureType.phase_shift:
                self.tuned_history["Kvp"]["phase shift"] = self.phase_shift
        elif self.tune_mode == ServoGain.Velocity.value.ki:
            self.tuned_history["Kvi"]["gain"] = self.gain
            if self.feature_type == FeatureType.magnitude_deviation:
                self.tuned_history["Kvi"]["magnitude deviation"] = self.magnitude_deviation
            elif self.feature_type == FeatureType.phase_shift:
                self.tuned_history["Kvi"]["phase shift"] = self.phase_shift

    def tune_loop(self, tune_loop: Loop_Type, ki_min: float, ki_max: float, feature_type: FeatureType):
        self.tune_loop_type = tune_loop
        self.max_gain = k_max
        self.min_gain = k_min
        self.feature_type = feature_type
        
        self.magnitude_deviation.append(self.compute_magnitude_deviation())
        self.phase_shift.append(self.compute_phase_shift())

        self.gain.append(self.rt605.joints[self.lag_joint].get_PID(tune_mode))

        self.print_gain(iter=0)

        # Compute initial gain
        if self.feature_type == FeatureType.magnitude_deviation:
            gain_init = k_min + self.magnitude_deviation[0]*(k_max - k_min) / (self.max_magnitude_deviation - self.min_magnitude_deviation)
        else:
            gain_init = k_min + self.phase_shift[0]*(k_max - k_min) / (self.max_phase_shift - self.min_phase_shift)

        if gain_init < k_min:
            gain_init = k_min
        elif gain_init > k_max:
            gain_init = k_max

        self.gain.append(gain_init)
        self.rt605.setPID(self.lag_joint, tune_mode, gain_init)
        self.rt605.run_HRSS_intp()

        if self.feature_type == FeatureType.magnitude_deviation:
            self.magnitude_deviation.append(self.compute_magnitude_deviation())
        else:
            self.phase_shift.append(self.compute_phase_shift())

        self.print_gain(iter=1)

        if self.feature_type == FeatureType.magnitude_deviation:
            self.magnitude_deviation.append(self.compute_magnitude_deviation())
        else:
            self.phase_shift.append(self.compute_phase_shift())

        self.print_gain(iter=1)

        # Start lagrange interpolation
        for iter in range(2, self.iter):
            if self.feature_type == FeatureType.magnitude_deviation:
                next_gain = self.lagrange_intp(self.gain, self.magnitude_deviation, k_min, k_max, 10000)
                self.gain.append(next_gain)
            
                self.rt605.setPID(self.lag_joint, tune_mode, next_gain)
                self.rt605.run_HRSS_intp()
            
                self.magnitude_deviation.append(self.compute_magnitude_deviation())
            else:
                next_gain = self.lagrange_intp(self.gain, self.phase_shift, k_min, k_max, 10000)
                self.gain.append(next_gain)
            
                self.rt605.setPID(self.lag_joint, tune_mode, next_gain)
                self.rt605.run_HRSS_intp()
            
                self.phase_shift.append(self.compute_phase_shift())
            self.print_gain(iter)
            # self.rt605.plot_joint(True)
            if self.feature_type == FeatureType.magnitude_deviation:
                next_gain = self.lagrange_intp(self.gain, self.magnitude_deviation, k_min, k_max, 10000)
                self.gain.append(next_gain)
            
                self.rt605.setPID(self.lag_joint, tune_mode, next_gain)
                self.rt605.run_HRSS_intp()
            
                self.magnitude_deviation.append(self.compute_magnitude_deviation())
            else:
                next_gain = self.lagrange_intp(self.gain, self.phase_shift, k_min, k_max, 10000)
                self.gain.append(next_gain)
            
                self.rt605.setPID(self.lag_joint, tune_mode, next_gain)
                self.rt605.run_HRSS_intp()
            
                self.phase_shift.append(self.compute_phase_shift())
            self.print_gain(iter)
        
    def plot3D(self):
        pass
        
        
        
    
    def reset_RT605(self):
        #self.rt605.resetPID()
        self.phase_shift = []
        self.magnitude_deviation = []
        self.features = []
        self.gain = []
        self.kpi = []
        self.kpp = []
        self.kvi = []
        self.kvp = []

    def print_gain(self,iter:int):
        if self.feature_type == FeatureType.magnitude_deviation:
            if self.tune_mode == ServoGain.Position.value.kp:
                print(f"iter: {iter} || kpp: {self.gain[iter]} || magnitude_deviation: {self.magnitude_deviation[iter]}")
            elif self.tune_mode == ServoGain.Position.value.ki:
                print(f"iter: {iter} || kpi: {self.gain[iter]} || magnitude_deviation: {self.magnitude_deviation[iter]}")
            elif self.tune_mode == ServoGain.Velocity.value.kp:
                print(f"iter: {iter} || kvp: {self.gain[iter]} || magnitude_deviation: {self.magnitude_deviation[iter]}")
            elif self.tune_mode == ServoGain.Velocity.value.ki:
                print(f"iter: {iter} || kvi: {self.gain[iter]} || magnitude_deviation: {self.magnitude_deviation[iter]}")
        else:
            if self.tune_mode == ServoGain.Position.value.kp:
                print(f"iter: {iter} || kpp: {self.gain[iter]} || phase_shift: {self.phase_shift[iter]}")
            elif self.tune_mode == ServoGain.Position.value.ki:
                print(f"iter: {iter} || kpi: {self.gain[iter]} || phase_shift: {self.phase_shift[iter]}")
            elif self.tune_mode == ServoGain.Velocity.value.kp:
                print(f"iter: {iter} || kvp: {self.gain[iter]} || phase_shift: {self.phase_shift[iter]}")
            elif self.tune_mode == ServoGain.Velocity.value.ki:
                print(f"iter: {iter} || kvi: {self.gain[iter]} || phase_shift: {self.phase_shift[iter]}")

    def save_json_file(self):
        """
        save Kpp, Kpi, Kvp, Kvi to json file
        """
        # if self.feature_type == FeatureType.magnitude_deviation:
        #     if self.tune_mode == ServoGain.Position.value.kp:
        #         save_file_path = f"kpp_iter_{self.iter}_max_{self.max_gain}_mag.json"
        #     elif self.tune_mode == ServoGain.Position.value.ki:
        #         save_file_path = f"kpi_iter_{self.iter}_max_{self.max_gain}_mag.json"
        #     elif self.tune_mode == ServoGain.Velocity.value.kp:
        #         save_file_path = f"kvp_iter_{self.iter}_max_{self.max_gain}_mag.json"
        #     elif self.tune_mode == ServoGain.Velocity.value.ki:
        #         save_file_path = f"kvi_iter_{self.iter}_max_{self.max_gain}_mag.json"
        # else:
        #     if self.tune_mode == ServoGain.Position.value.kp:
        #         save_file_path = f"kpp_iter_{self.iter}_max_{self.max_gain}_phase.json"
        #     elif self.tune_mode == ServoGain.Position.value.ki:
        #         save_file_path = f"kpi_iter_{self.iter}_max_{self.max_gain}_phase.json"
        #     elif self.tune_mode == ServoGain.Velocity.value.kp:
        #         save_file_path = f"kvp_iter_{self.iter}_max_{self.max_gain}_phase.json"
        #     elif self.tune_mode == ServoGain.Velocity.value.ki:
        #         save_file_path = f"kvi_iter_{self.iter}_max_{self.max_gain}_phase.json"
        save_file_path = "test3.json"
        
        # Writing the dictionary to a JSON file
        with open(save_file_path, 'w') as json_file:
            json.dump(self.tuned_history, json_file, indent=4)

        # with open(save_file_path, 'w', newline='') as csvfile:
        #     csvwriter = csv.writer(csvfile)
        #     # Write the header
        #     header = []
        #     header.append("kpp")
        #     if self
        #     if self.tune_mode == Servo
        #     csvwriter.writerow(['', 'y'])
        #     # Write the data
        #     for x, y in zip(self.gain, self.magnitude_deviation):
        #         csvwriter.writerow([x, y])

        print(f"Data successfully written to {save_file_path}")
    
    def save_csv_file(self):
        if self.feature_type == FeatureType.magnitude_deviation:
            if self.tune_mode == ServoGain.Position.value.kp:
                save_file_path = f"kpp_iter_{self.iter}_max_{self.max_gain}_mag.csv"
            elif self.tune_mode == ServoGain.Position.value.ki:
                save_file_path = f"kpi_iter_{self.iter}_max_{self.max_gain}_mag.csv"
            elif self.tune_mode == ServoGain.Velocity.value.kp:
                save_file_path = f"kvp_iter_{self.iter}_max_{self.max_gain}_mag.csv"
            elif self.tune_mode == ServoGain.Velocity.value.ki:
                save_file_path = f"kvi_iter_{self.iter}_max_{self.max_gain}_mag.csv"
        else:
            if self.tune_mode == ServoGain.Position.value.kp:
                save_file_path = f"kpp_iter_{self.iter}_max_{self.max_gain}_phase.csv"
            elif self.tune_mode == ServoGain.Position.value.ki:
                save_file_path = f"kpi_iter_{self.iter}_max_{self.max_gain}_phase.csv"
            elif self.tune_mode == ServoGain.Velocity.value.kp:
                save_file_path = f"kvp_iter_{self.iter}_max_{self.max_gain}_phase.csv"
            elif self.tune_mode == ServoGain.Velocity.value.ki:
                save_file_path = f"kvi_iter_{self.iter}_max_{self.max_gain}_phase.csv"

    

        with open(save_file_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            # Write the header
            csvwriter.writerow(['x', 'y'])
            # Write the data
            if self.feature_type == FeatureType.magnitude_deviation:
                for x, y in zip(self.gain, self.magnitude_deviation):
                    csvwriter.writerow([x, y])
            else:
                for x, y in zip(self.gain, self.phase_shift):
                    csvwriter.writerow([x, y])
        print(f"Data successfully written to {save_file_path}")
    
    def lagrange_2D(self):
        pass

    def plot_result(self):
        self.rt605.plot_joint()
        self.rt605.plot_error()
        self.rt605.freq_response()

def main():
    interpolation = Intp(iter=20)
    interpolation.inference()

    interpolation.run(ServoGain.Velocity.value.ki, 0.1, 10, FeatureType.magnitude_deviation)
    interpolation.reset_RT605()

    interpolation.run(ServoGain.Velocity.value.kp, 1, 100, FeatureType.magnitude_deviation)
    interpolation.reset_RT605()

    interpolation.run(ServoGain.Position.value.ki, 0.1, 10, FeatureType.magnitude_deviation)
    interpolation.reset_RT605()

    interpolation.run(ServoGain.Position.value.kp, 1, 100, FeatureType.magnitude_deviation)
    interpolation.save_json_file()
    # interpolation.reset_RT605()

    # for iter in range(0, 1):
    
    #     interpolation.inference()
        
    #     interpolation.run(ServoGain.Velocity.value.kp, 0.01, 100)


    #     interpolation.run(ServoGain.Velocity.value.ki, 0.001, 50)

    #     interpolation.run(ServoGain.Position.value.kp, 0.01, 100)

    #     # interpolation.run(ServoGain.Position.value.ki, 0.001, 50)

    #     interpolation.plot_result()
    
    # interpolation.save_file()

def test(tune_mode:ServoGain, min, max, feature_type:FeatureType):
    interpolation = Intp(iter=10)
    interpolation.inference()

    interpolation.run(tune_mode, min, max, feature_type)
    interpolation.save_csv_file()


if __name__ == "__main__":
    # test(ServoGain.Position.value.kp, 1, 100, FeatureType.magnitude_deviation)
    # test(ServoGain.Position.value.kp, 1, 100, FeatureType.phase_shift)
    # test(ServoGain.Position.value.ki, 0.05, 10, FeatureType.magnitude_deviation)
    # test(ServoGain.Position.value.ki, 0.05, 10, FeatureType.phase_shift)
    # test(ServoGain.Velocity.value.kp, 5, 100, FeatureType.magnitude_deviation)
    # test(ServoGain.Velocity.value.kp, 5, 100, FeatureType.phase_shift)
    # test(ServoGain.Velocity.value.ki, 0.08, 10, FeatureType.magnitude_deviation)
    # test(ServoGain.Velocity.value.ki, 0.08, 10, FeatureType.phase_shift)
    main()