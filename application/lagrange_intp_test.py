import sys
import numpy as np
from scipy.optimize import least_squares
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import csv
import json
from scipy.interpolate import lagrange
import os
sys.path.append('..')
#sys.path.insert(1,"../rt605")
sys.path.append("../mismatch_classification/")

from rt605 import RT605
from libs.ServoMotor import ServoMotor
from libs.type_define import*

from mismatch_classification.model_playground import *

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import datetime
from tqdm import tqdm
import copy

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

    

    kpp: list
    kpi: list
    kvp: list
    kvi: list

    pointsCount: int

    feature: list
    max_feature: float
    min_feature: float

    max_gain: float
    min_gain: float

    iter:int
    tune_mode: ServoGain
    
    gain: list
    bandwidth: list

    save_file: str

    tuned_history: dict

    tune_loop_mode: bool
    tune_loop_type: Loop_Type

    num_of_test: int

    tracking_err_before: list
    tracking_err_after: list

    def __init__(self, iter:int=10, pointsCount:int=1000):
        self.rt605 = RT605()
        self.trajectory_path = "../data/Path/"+"sine_f6_full_joints.txt"

        self.rt605.load_HRSS_trajectory(self.trajectory_path)
        self.rt605.compute_GTorque.enable_Gtorq(en=True)
        self.rt605.compute_friction.enable_friction(en=True)


        # Run model inference: get the initial lag joints values
        # Initial mismatch classification model
        

        self.max_magnitude_deviation = 10
        self.min_magnitude_deviation = 0
        self.max_phase_shift = 5
        self.min_phase_shift = -0.1

        self.iter = iter

        self.magnitude_deviation = []
        self.magnitude_deviation = []
        self.phase_shift = []
        self.gain = []
        self.bandwidth = []

        self.kpp = []
        self.kpi = []
        self.kvp = []
        self.kvi = []

        self.features = []
        self.max_feature = 2
        self.min_feature = -2
        
        self.max_gain = 0.0
        self.min_gain = 0.0

        self.feature_type = None

        self.tuned_history = {}
        self.tuned_history["iterations"] = {}
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

        self.tune_loop_type = Loop_Type.vel
        self.tune_loop_mode = False
    
        self.pointsCount = pointsCount

        self.num_of_test = 0

        self.min_feature = 1e-10
        



        

    def inference(self)->None:
        
        self.rt605.run_HRSS_intp()
        input_data = self.rt605.q_pos_err
        temp = [3,4,1,2,5,0]

        input_data = torch.from_numpy(input_data).float()
        input_data = input_data.permute(1, 0).unsqueeze(0)
        self.model_weight_path = '6_12_17_23_best_model_acc_95.875.pth'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
        input_shape = input_data.shape[1]
        self.model = CNN1D(input_dim=input_shape, output_dim=6).to(self.device)
        self.model.load_state_dict(torch.load(self.model_weight_path))
        self.model.eval()
        input_data = input_data.to(self.device)

        with torch.no_grad():
            output = self.model(input_data)
        
        self.lag_joint = torch.argmax(output).item()
        self.lag_joint = temp[self.num_of_test]
        self.num_of_test = self.num_of_test+1
        print(f"Lag joint: {self.lag_joint}")
        # self.rt605.plot_error()
    
    def lagrange_2D(self, x_data, y_data, z_data, min_x:float, max_x:float, min_y:float, max_y:float, plot=False)->tuple:
        """
        input:
        x_data, y_data --> kp, ki
        z_data --> gain/phase feature
        min/max_x: min/max value of kp
        min/max_y: min/max value of ki
        pointcount: interpolation resolution

        return:
        optimal kp and ki
        """
        

        result = 0
        min_z = np.inf
        def lagrange_basis_polynomial(x, x_points, i):
            basis = 1
            for j in range(len(x_points)):
                if i != j and (x_points[i] != x_points[j]):
                    basis *= (x - x_points[j]) / (x_points[i] - x_points[j])
            return basis
        
        opt_x, opt_y = 0.0,0.0

        if plot:    
            f_points = np.zeros((self.pointsCount, self.pointsCount))

        for idx_x in tqdm(range(self.pointsCount)):
            x = min_x + idx_x * (max_x - min_x)/self.pointsCount

            for idx_y in range(self.pointsCount):
                y = min_y + idx_y * (max_y - min_y)/self.pointsCount
                
                for i in range(len(x_data)):
                    for j in range(len(y_data)):
                        L_i = lagrange_basis_polynomial(x, x_data, i)
                        M_j = lagrange_basis_polynomial(y, y_data, j)
                        result += z_data[i][j] * L_i * M_j
                if plot:
                    f_points[idx_x][idx_y] = result

                if abs(result) < min_z:
                    min_z = abs(result)
                    opt_x = x
                    opt_y = y
        fig = plt.figure()

        if plot:

            ax = fig.add_subplot(111, projection='3d')
            x = np.linspace(min_x, max_x, self.pointsCount)
            y = np.linspace(min_y, max_y, self.pointsCount)
            x_grid, y_grid = np.meshgrid(x,y)
            surf = ax.plot_surface(x_grid, y_grid, f_points, cmap='viridis', edgecolor='none')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Feature')
            fig.colorbar(surf)
            plt.show()
        else:
            plt.close(fig)
        
        return opt_x, opt_y, fig

    def lagrange_intp(self, x_data, y_data, min_x, max_x, pointsCount:int=1000)->float:
        opt_gain = 0.0
        min_y = np.inf
        self.max_gain = max_x
        self.min_gain = min_x
        self.max_gain = max_x
        self.min_gain = min_x
        self.pointsCount = pointsCount

        for idx in tqdm(range(self.pointsCount)):
            x = min_x + idx * (max_x - min_x)/self.pointsCount
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
    
    # def run(self, tune_mode: ServoGain, k_min:float, k_max:float, feature_type: FeatureType=FeatureType.magnitude_deviation):
    #     self.tune_mode = tune_mode
    #     self.max_gain = k_max
    #     self.min_gain = k_min
    #     self.feature_type = feature_type
        
    #     self.magnitude_deviation.append(self.compute_magnitude_deviation())
    #     self.phase_shift.append(self.compute_phase_shift())
    
    def tune_gain(self, tune_mode: ServoGain, k_min:float, k_max:float, feature_type: FeatureType=FeatureType.magnitude_deviation):
        self.tune_mode = tune_mode
        self.max_gain = k_max
        self.min_gain = k_min
        self.feature_type = feature_type
        print(f"lag joint: {self.lag_joint}")
        self.magnitude_deviation.append(self.compute_magnitude_deviation())
        self.phase_shift.append(self.compute_phase_shift())
        self.rt605.freq_response(show=False)
        self.bandwidth.append(self.rt605.bandwidth[self.lag_joint])
        self.gain.append(self.rt605.joints[self.lag_joint].get_PID(tune_mode))

        self.print_gain(iter=0)
        self.rt605.freq_response(show=False)
        self.bandwidth.append(self.rt605.bandwidth[self.lag_joint])

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

        self.rt605.freq_response(show=False)
        self.bandwidth.append(self.rt605.bandwidth[self.lag_joint])

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

            self.rt605.freq_response(show=False)
            self.bandwidth.append(self.rt605.bandwidth[self.lag_joint])

            self.print_gain(iter)

            fig_file_name = os.path.join("tune_gain_history",f'{iter}.png')
            if self.feature_type == FeatureType.magnitude_deviation:
                self.plot_lagrane_interpolation(self.gain, self.magnitude_deviation, k_min, k_max, 10000, fig_file_name)
            else:
                self.plot_lagrane_interpolation(self.gain, self.phase_shift, k_min, k_max, 10000, fig_file_name)
            # # self.rt605.plot_joint(True)
            # if self.feature_type == FeatureType.magnitude_deviation:
            #     next_gain = self.lagrange_intp(self.gain, self.magnitude_deviation, k_min, k_max, 10000)
            #     self.gain.append(next_gain)
            
            #     self.rt605.setPID(self.lag_joint, tune_mode, next_gain)
            #     self.rt605.run_HRSS_intp()
            
            #     self.magnitude_deviation.append(self.compute_magnitude_deviation())
            # else:
            #     next_gain = self.lagrange_intp(self.gain, self.phase_shift, k_min, k_max, 10000)
            #     self.gain.append(next_gain)
            
            #     self.rt605.setPID(self.lag_joint, tune_mode, next_gain)
            #     self.rt605.run_HRSS_intp()
            
            #     self.phase_shift.append(self.compute_phase_shift())
            # self.print_gain(iter)
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
            self.tuned_history["Kpp"]["bandwidth"] = self.bandwidth
        elif self.tune_mode == ServoGain.Position.value.ki:
            self.tuned_history["Kpi"]["gain"] = self.gain
            if self.feature_type == FeatureType.magnitude_deviation:
                self.tuned_history["Kpi"]["magnitude deviation"] = self.magnitude_deviation
            elif self.feature_type == FeatureType.phase_shift:
                self.tuned_history["Kpi"]["phase shift"] = self.phase_shift
            self.tuned_history["Kpi"]["bandwidth"] = self.bandwidth
        elif self.tune_mode == ServoGain.Velocity.value.kp:
            self.tuned_history["Kvp"]["gain"] = self.gain
            if self.feature_type == FeatureType.magnitude_deviation:
                self.tuned_history["Kvp"]["magnitude deviation"] = self.magnitude_deviation
            elif self.feature_type == FeatureType.phase_shift:
                self.tuned_history["Kvp"]["phase shift"] = self.phase_shift
            self.tuned_history["Kvp"]["bandwidth"] = self.bandwidth
        elif self.tune_mode == ServoGain.Velocity.value.ki:
            self.tuned_history["Kvi"]["gain"] = self.gain
            if self.feature_type == FeatureType.magnitude_deviation:
                self.tuned_history["Kvi"]["magnitude deviation"] = self.magnitude_deviation
            elif self.feature_type == FeatureType.phase_shift:
                self.tuned_history["Kvi"]["phase shift"] = self.phase_shift
            self.tuned_history["Kvi"]["bandwidth"] = self.bandwidth

        

    def tune_loop(self, tune_loop: Loop_Type, kp_min: float, kp_max: float, ki_min: float, ki_max: float, feature_type: FeatureType):
        self.tune_loop_mode = True
        self.tune_loop_type = tune_loop
        self.feature_type = feature_type

        self.magnitude_deviation = [[0 for _ in range(self.iter)] for _ in range(self.iter)]
        self.phase_shift = [[0 for _ in range(self.iter)] for _ in range(self.iter)]
        self.bandwidth = [[0 for _ in range(self.iter)] for _ in range(self.iter)]

        self.magnitude_deviation[0][0] = self.compute_magnitude_deviation()
        self.phase_shift[0][0] = self.compute_phase_shift()

        self.kpp.append(self.rt605.joints[self.lag_joint].get_PID(ServoGain.Position.value.kp))
        self.kpi.append(self.rt605.joints[self.lag_joint].get_PID(ServoGain.Position.value.ki))
        self.kvp.append(self.rt605.joints[self.lag_joint].get_PID(ServoGain.Velocity.value.kp))
        self.kvi.append(self.rt605.joints[self.lag_joint].get_PID(ServoGain.Velocity.value.ki))

        self.print_gain(iter=0, tune_loop=True)
        self.rt605.sweep(show=False)
        print(self.rt605.bandwidth[self.lag_joint])
        self.bandwidth[0][0] = self.rt605.bandwidth[self.lag_joint]

        # Compute initial gain
        if self.feature_type == FeatureType.magnitude_deviation:
            kp_init = kp_min + self.magnitude_deviation[0][0]*(kp_max - kp_min) / (self.max_magnitude_deviation - self.min_magnitude_deviation)
            ki_init = ki_min + self.magnitude_deviation[0][0]*(ki_max - ki_min) / (self.max_magnitude_deviation - self.min_magnitude_deviation)
        else:
            kp_init = kp_min + self.phase_shift[0]*(kp_max - kp_min) / (self.max_phase_shift - self.min_phase_shift)
            ki_init = ki_min + self.phase_shift[0]*(ki_max - ki_min) / (self.max_phase_shift - self.min_phase_shift)
            
        kp_init = max(kp_min, min(kp_max, kp_init))
        ki_init = min(ki_min, min(ki_min, ki_init))

        if tune_loop == Loop_Type.pos:
            self.kpp.append(kp_init)
            self.kpi.append(ki_init)
            self.rt605.setPID(self.lag_joint, ServoGain.Position.value.kp, kp_init)
            self.rt605.run_HRSS_intp()
            self.magnitude_deviation[0][1] = self.compute_magnitude_deviation()
            # self.rt605.sweep(show=False)
            # self.bandwidth[0][1] = self.rt605.bandwidth[self.lag_joint]

            # reset to previous gain
            self.rt605.setPID(self.lag_joint, ServoGain.Position.value.kp, self.kpp[0])

            self.rt605.setPID(self.lag_joint, ServoGain.Position.value.ki, ki_init)
            self.rt605.run_HRSS_intp()
            self.magnitude_deviation[1][0] = self.compute_magnitude_deviation()
            # self.rt605.sweep(show=False)
            # self.bandwidth[1][0] = self.rt605.bandwidth[self.lag_joint]
            
            self.rt605.setPID(self.lag_joint, ServoGain.Position.value.kp, kp_init)
            self.rt605.run_HRSS_intp()
            self.magnitude_deviation[1][1] = self.compute_magnitude_deviation()
            # self.rt605.sweep(show=False)
            # self.bandwidth[1][1] = self.rt605.bandwidth[self.lag_joint]

            
        elif tune_loop == Loop_Type.vel:
            # self.rt605.setPID(self.lag_joint, ServoGain.Velocity.value.kp, kp_init)
            # self.rt605.setPID(self.lag_joint, ServoGain.Velocity.value.ki, ki_init)
            self.kvp.append(kp_init)
            self.kvi.append(ki_init)

            self.rt605.setPID(self.lag_joint, ServoGain.Velocity.value.kp, kp_init)
            self.rt605.run_HRSS_intp()
            self.magnitude_deviation[0][1] = self.compute_magnitude_deviation()
            # self.rt605.sweep(show=False)
            # self.bandwidth[0][1] = self.rt605.bandwidth[self.lag_joint]

            # reset to previous gain
            self.rt605.setPID(self.lag_joint, ServoGain.Velocity.value.kp, self.kvp[0])

            self.rt605.setPID(self.lag_joint, ServoGain.Velocity.value.ki, ki_init)
            self.rt605.run_HRSS_intp()
            self.magnitude_deviation[1][0] = self.compute_magnitude_deviation()
            # self.rt605.sweep(show=False)
            # self.bandwidth[1][0] = self.rt605.bandwidth[self.lag_joint]
            
            self.rt605.setPID(self.lag_joint, ServoGain.Velocity.value.kp, kp_init)
            self.rt605.run_HRSS_intp()
            self.magnitude_deviation[1][1] = self.compute_magnitude_deviation()
            # self.rt605.sweep(show=False)
            # self.bandwidth[1][1] = self.rt605.bandwidth[self.lag_joint]

        # if self.feature_type == FeatureType.magnitude_deviation:
        #     self.magnitude_deviation.append(self.compute_magnitude_deviation())
        # else:
        #     self.phase_shift.append(self.compute_phase_shift())

        self.print_gain(iter=1, tune_loop=True)

        # self.rt605.sweep(show=False)
        # self.bandwidth.append(self.rt605.bandwidth[self.lag_joint])

        # Start lagrange interpolation
        for iter in range(2, self.iter):
            if self.feature_type == FeatureType.magnitude_deviation:
                if tune_loop == Loop_Type.pos:
                    if iter != self.iter-1:
                        kp_next, ki_next, _ = self.lagrange_2D(self.kpp, self.kpi, self.magnitude_deviation, kp_min, kp_max, ki_min, ki_max, plot=False)
                    else:
                        kp_next, ki_next, fig = self.lagrange_2D(self.kpp, self.kpi, self.magnitude_deviation, kp_min, kp_max, ki_min, ki_max, plot=True)

                    self.kpp.append(kp_next)
                    self.kpi.append(ki_next)
                
                    self.rt605.setPID(self.lag_joint, ServoGain.Position.value.kp, kp_next)
                    for i, ki in enumerate(self.kpi):
                        self.rt605.setPID(self.lag_joint, ServoGain.Position.value.ki, ki)
                        self.rt605.run_HRSS_intp()
                        self.magnitude_deviation[iter][i] = self.compute_magnitude_deviation()
                        # self.rt605.sweep(show=False)
                        # self.bandwidth[iter][i] = self.rt605.bandwidth[self.lag_joint]


                    self.rt605.setPID(self.lag_joint, ServoGain.Position.value.ki, ki_next)
                    for i, kp in enumerate(self.kpp):
                        self.rt605.setPID(self.lag_joint, ServoGain.Position.value.kp, kp)
                        self.rt605.run_HRSS_intp()
                        self.magnitude_deviation[i][iter] = self.compute_magnitude_deviation()
                        # self.rt605.sweep(show=False)
                        # self.bandwidth[i][iter] = self.rt605.bandwidth[self.lag_joint]

                elif tune_loop == Loop_Type.vel:
                    if iter != self.iter-1:
                        kp_next, ki_next, _ = self.lagrange_2D(self.kvp, self.kvi, self.magnitude_deviation, kp_min, kp_max, ki_min, ki_max, plot=False)
                    else:
                        kp_next, ki_next, fig = self.lagrange_2D(self.kvp, self.kvi, self.magnitude_deviation, kp_min, kp_max, ki_min, ki_max, plot=True)
                    self.kvp.append(kp_next)
                    self.kvi.append(ki_next)
                
                    self.rt605.setPID(self.lag_joint, ServoGain.Velocity.value.kp, kp_next)
                    for i, ki in enumerate(self.kvi):
                        self.rt605.setPID(self.lag_joint, ServoGain.Velocity.value.ki, ki)
                        self.rt605.run_HRSS_intp()
                        self.magnitude_deviation[iter][i] = self.compute_magnitude_deviation()
                        # self.rt605.sweep(show=False)
                        # self.bandwidth[iter][i] = self.rt605.bandwidth[self.lag_joint]


                    self.rt605.setPID(self.lag_joint, ServoGain.Velocity.value.ki, ki_next)
                    for i, kp in enumerate(self.kvp):
                        self.rt605.setPID(self.lag_joint, ServoGain.Velocity.value.kp, kp)
                        self.rt605.run_HRSS_intp()
                        self.magnitude_deviation[i][iter] = self.compute_magnitude_deviation()
                        # self.rt605.sweep(show=False)
                        # self.bandwidth[i][iter] = self.rt605.bandwidth[self.lag_joint]
            else:
                if tune_loop == Loop_Type.pos:
                    if iter != self.iter-1:
                        kp_next, ki_next, _ = self.lagrange_2D(self.kpp, self.kpi, self.phase_shift, kp_min, kp_max, ki_min, ki_max, plot=False)
                    else:
                        kp_next, ki_next, fig = self.lagrange_2D(self.kpp, self.kpi, self.phase_shift, kp_min, kp_max, ki_min, ki_max, plot=True)
                    self.kpp.append(kp_next)
                    self.kpi.append(ki_next)
                
                    self.rt605.setPID(self.lag_joint, ServoGain.Position.value.kp, kp_next)
                    for i, ki in enumerate(self.kpi):
                        self.rt605.setPID(self.lag_joint, ServoGain.Position.value.ki, ki)
                        self.rt605.run_HRSS_intp()
                        self.phase_shift[iter][i] = self.compute_phase_shift()
                        # self.rt605.sweep(show=False)
                        # self.bandwidth[iter][i] = self.rt605.bandwidth[self.lag_joint]


                    self.rt605.setPID(self.lag_joint, ServoGain.Position.value.ki, ki_next)
                    for i, kp in enumerate(self.kpp):
                        self.rt605.setPID(self.lag_joint, ServoGain.Position.value.kp, kp)
                        self.rt605.run_HRSS_intp()
                        self.phase_shift[i][iter] = self.compute_phase_shift()
                        # self.rt605.sweep(show=False)
                        # self.bandwidth[i][iter] = self.rt605.bandwidth[self.lag_joint]

                elif tune_loop == Loop_Type.vel:
                    if iter != self.iter:
                        kp_next, ki_next, _ = self.lagrange_2D(self.kvp, self.kvi, self.phase_shift, kp_min, kp_max, ki_min, ki_max, plot=False)
                    else:
                        kp_next, ki_next, fig = self.lagrange_2D(self.kvp, self.kvi, self.phase_shift, kp_min, kp_max, ki_min, ki_max, plot=True)
                    self.kvp.append(kp_next)
                    self.kvi.append(ki_next)
                
                    self.rt605.setPID(self.lag_joint, ServoGain.Velocity.value.kp, kp_next)
                    for i, ki in enumerate(self.kvi):
                        self.rt605.setPID(self.lag_joint, ServoGain.Velocity.value.ki, ki)
                        self.rt605.run_HRSS_intp()
                        self.phase_shift[iter][i] = self.compute_phase_shift()
                        # self.rt605.sweep(show=False)
                        # self.bandwidth[iter][i] = self.rt605.bandwidth[self.lag_joint]


                    self.rt605.setPID(self.lag_joint, ServoGain.Velocity.value.ki, ki_next)
                    for i, kp in enumerate(self.kvp):
                        self.rt605.setPID(self.lag_joint, ServoGain.Velocity.value.kp, kp)
                        self.rt605.run_HRSS_intp()
                        self.phase_shift[i][iter] = self.compute_phase_shift()
                        # self.rt605.sweep(show=False)
                        # self.bandwidth[i][iter] = self.rt605.bandwidth[self.lag_joint]

            self.print_gain(iter, tune_loop=True)

            # self.rt605.sweep(show=False)
            # self.bandwidth.append(self.rt605.bandwidth[self.lag_joint])

        fig.show()

        if self.tune_loop_type == Loop_Type.pos:
            self.tuned_history["Kpp"]["bandwidth"] = self.bandwidth
            self.tuned_history["Kpi"]["bandwidth"] = self.bandwidth
            self.tuned_history["Kpp"]["gain"] = self.kpp
            self.tuned_history["Kpi"]["gain"] = self.kpi
            if self.feature_type == FeatureType.magnitude_deviation:
                self.tuned_history["Kpp"]["magnitude deviation"] = self.magnitude_deviation
                self.tuned_history["Kpi"]["magnitude deviation"] = self.magnitude_deviation
            elif self.feature_type == FeatureType.phase_shift:
                self.tuned_history["Kpp"]["phase shift"] = self.phase_shift
                self.tuned_history["Kpi"]["phase shift"] = self.phase_shift
        elif self.tune_loop_type == Loop_Type.vel:
            self.tuned_history["Kvp"]["bandwidth"] = self.bandwidth
            self.tuned_history["Kvi"]["bandwidth"] = self.bandwidth
            self.tuned_history["Kvp"]["gain"] = self.kvp
            self.tuned_history["Kvi"]["gain"] = self.kvi
            if self.feature_type == FeatureType.magnitude_deviation:
                self.tuned_history["Kvp"]["magnitude deviation"] = self.magnitude_deviation
                self.tuned_history["Kvi"]["magnitude deviation"] = self.magnitude_deviation
            elif self.feature_type == FeatureType.phase_shift:
                self.tuned_history["Kvp"]["phase shift"] = self.phase_shift
                self.tuned_history["Kvi"]["phase shift"] = self.phase_shift
    
    def polynomial_fit(self, x_data:list, y_data:list, x_min:float, x_max:float, pointsCount:int=1000)->float:
        
        coefficients = np.polyfit(x_data, y_data, len(x_data)-1)
        polynomial = np.poly1d(coefficients)

        # Generate x values for plotting the fitted polynomial
        x_fit = np.linspace(x_min, x_max, pointsCount)
        y_fit = polynomial(x_fit)

        #         # Plot the original points
        # plt.scatter(x_data, y_data, color='red', label='Original Points')

        # # Plot the fitted polynomial
        # plt.plot(x_fit, y_fit, color='blue', label=f'Fitted Polynomial: {polynomial}')

        # # Add labels and title
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.title('Polynomial Fit to Given Points')
        # plt.legend()
        # plt.grid(True)

        # # Show the plot
        # plt.show()

        return x_fit[np.argmin(y_fit)]

    # def tuner(self, iteration=6):
    #     interpolation = Intp(iter=8)
    #     tune_history = {}
    #     for iter in range(iteration):
    #         interpolation.inference()

    #         interpolation.tune_gain(ServoGain.Velocity.value.ki, 0.1, 200, FeatureType.magnitude_deviation)
    #         interpolation.reset_RT605()

    #         interpolation.tune_gain(ServoGain.Velocity.value.kp, 5, 100, FeatureType.magnitude_deviation)
    #         interpolation.reset_RT605()

    #         interpolation.tune_gain(ServoGain.Position.value.kp, 5, 150, FeatureType.phase_shift)
    #         interpolation.reset_RT605()

            
    #         tune_history[f"iter{iter}"]=interpolation.tuned_history
    #         tune_history[f"iter{iter}"]["iter"] = iter

    #     interpolation.save_json_file()
        
    def plot3D(self):
        if self.tune_loop_mode==False:
            return
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        mag_dev = np.array(self.magnitude_deviation)
        ph_shift = np.array(self.phase_shift)



        if self.tune_loop_type == Loop_Type.pos:

            p_gain, i_gain = np.meshgrid(self.kpp, self.kpi)
            if self.feature_type == FeatureType.magnitude_deviation:
                
                surf = ax.plot_surface(p_gain, i_gain, mag_dev, cmap=cm.viridis, edgecolor='none')
            else:
                surf = ax.plot_surface(p_gain, i_gain, ph_shift, cmap=cm.viridis, edgecolor='none')
        elif self.tune_loop_type == Loop_Type.vel:
            p_gain, i_gain = np.meshgrid(self.kvp, self.kvi)
            if self.feature_type == FeatureType.magnitude_deviation:
                surf = ax.plot_surface(p_gain, i_gain, mag_dev, cmap=cm.viridis, edgecolor='none')
            else:
                surf = ax.plot_surface(p_gain, i_gain, ph_shift, cmap=cm.viridis, edgecolor='none')



        # Add color bar which maps values to colors
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')

        plt.show()
    
    def reset_RT605(self):
        #self.rt605.resetPID()
        self.bandwidth = []
        self.phase_shift = []
        self.magnitude_deviation = []
        self.features = []
        self.gain = []
        self.kpi = []
        self.kpp = []
        self.kvi = []
        self.kvp = []

    def print_gain(self,iter:int, tune_loop:bool=False):
        if tune_loop:
            if self.tune_loop_type == Loop_Type.pos:
                if self.feature_type == FeatureType.magnitude_deviation:
                    print(f"iter: {iter} || kpp: {self.kpp[iter]} || BW: {self.rt605.bandwidth[iter]}")
                    print(f"iter: {iter} || kpi: {self.kpi[iter]} || BW: {self.rt605.bandwidth[iter]}")   
                    print(f'magnitude deviation:{self.magnitude_deviation}')       
                else:
                    print(f"iter: {iter} || kpp: {self.kpp[iter]} || BW: {self.rt605.bandwidth[iter]}")
                    print(f"iter: {iter} || kpi: {self.kpi[iter]} || BW: {self.rt605.bandwidth[iter]}")
                    print(f'magnitude deviation:{self.phase_shift}')  
            else:
                if self.feature_type == FeatureType.magnitude_deviation:
                    print(f"iter: {iter} || kvp: {self.kvp[iter]} || BW: {self.rt605.bandwidth[iter]}")
                    print(f"iter: {iter} || kvi: {self.kvi[iter]} || BW: {self.rt605.bandwidth[iter]}") 
                    print(f'magnitude deviation:{self.magnitude_deviation}')           
                else:
                    print(f"iter: {iter} || kvp: {self.kvp[iter]} || BW: {self.rt605.bandwidth[iter]}")
                    print(f"iter: {iter} || kvi: {self.kvi[iter]} || BW: {self.rt605.bandwidth[iter]}")
                    print(f'magnitude deviation:{self.phase_shift}')
        else:
            if self.feature_type == FeatureType.magnitude_deviation:
                if self.tune_mode == ServoGain.Position.value.kp:
                    print(f"iter: {iter} || kpp: {self.gain[iter]} || magnitude_deviation: {self.magnitude_deviation[iter]} || BW: {self.bandwidth[iter]}")
                elif self.tune_mode == ServoGain.Position.value.ki:
                    print(f"iter: {iter} || kpi: {self.gain[iter]} || magnitude_deviation: {self.magnitude_deviation[iter]} || BW: {self.bandwidth[iter]}")
                elif self.tune_mode == ServoGain.Velocity.value.kp:
                    print(f"iter: {iter} || kvp: {self.gain[iter]} || magnitude_deviation: {self.magnitude_deviation[iter]} || BW: {self.bandwidth[iter]}")
                elif self.tune_mode == ServoGain.Velocity.value.ki:
                    print(f"iter: {iter} || kvi: {self.gain[iter]} || magnitude_deviation: {self.magnitude_deviation[iter]} || BW: {self.bandwidth[iter]}")
            else:
                if self.tune_mode == ServoGain.Position.value.kp:
                    print(f"iter: {iter} || kpp: {self.gain[iter]} || phase_shift: {self.phase_shift[iter]} || BW: {self.bandwidth[iter]}")
                elif self.tune_mode == ServoGain.Position.value.ki:
                    print(f"iter: {iter} || kpi: {self.gain[iter]} || phase_shift: {self.phase_shift[iter]} || BW: {self.bandwidth[iter]}")
                elif self.tune_mode == ServoGain.Velocity.value.kp:
                    print(f"iter: {iter} || kvp: {self.gain[iter]} || phase_shift: {self.phase_shift[iter]} || BW: {self.bandwidth[iter]}")
                elif self.tune_mode == ServoGain.Velocity.value.ki:
                    print(f"iter: {iter} || kvi: {self.gain[iter]} || phase_shift: {self.phase_shift[iter]} || BW: {self.bandwidth[iter]}")

    def save_json_file(self, tune_history:dict=None):
        """
        save Kpp, Kpi, Kvp, Kvi to json file
        """

        now = datetime.datetime.now()
        timestamp = now.strftime("%m_%d_%H_%M")

        save_file_path = f"{timestamp}.json"
        
        # Writing the dictionary to a JSON file
        with open(save_file_path, 'w') as json_file:
            if(tune_history is not None):
                json.dump(tune_history, json_file, indent=4)
            else:
                json.dump(self.tuned_history, json_file, indent=4)

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

    def plot_lagrane_interpolation(self, x_data, y_data, min_x, max_x, pointsCount=10000, save_file_path:str=None):

        # poly = lagrange(x_data, y_data)

        min_x = min(x_data)
        max_x = max(x_data)


        x_interp = np.linspace(min_x, max_x, pointsCount)
        # y_interp = poly(x_interp)
        y_interp = np.zeros(x_interp.shape)


        for idx in tqdm(range(self.pointsCount)):
            # x = min_x + idx * (max_x - min_x)/self.pointsCount
            # y = 0.0

            # compute y value of polynomial(x)
            for i in range(len(x_data)):
                term = y_data[i]
                for j in range(len(x_data)):
                    if (j != i) and (x_data[i] != x_data[j]):
                        term = term * (x_interp[idx] - x_data[j])/(x_data[i] - x_data[j])
                y_interp[idx] = y_interp[idx] + term

        plt.figure(figsize=(10, 6))
        plt.plot(x_data, y_data, 'o', label='Original')
        plt.plot(x_interp, y_interp, '-', label='Lagrange Interpolation')

        zero_crossing_y = np.inf
        zero_crossing_x = np.inf
        for i, y in enumerate(y_interp):
            if abs(y)<zero_crossing_y:
                zero_crossing_x = x_interp[i]
                zero_crossing_y = abs(y)
        plt.plot(zero_crossing_x, zero_crossing_y, '*', label='solution')
        # Add labels and title
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Lagrange Interpolation')

        # Display the plot
        plt.legend()
        plt.grid(True)
        plt.savefig(save_file_path)
        plt.close()
    
    def plot_result(self):
        self.rt605.plot_joint()
        self.rt605.plot_error()
        self.rt605.freq_response()

    
        
    def plot_tune_history(self):
   
        bandwidth = []
        kpp = None
        kpi = None
        kvp = None
        kvi = None

        if "phase shift" in self.tuned_history["Kvi"]:
            kvi_feature = self.tuned_history["Kvi"]["phase shift"]
        elif "magnitude deviation" in self.tuned_history["Kvi"]:
            kvi_feature = self.tuned_history["Kvi"]["magnitude deviation"]
        else:
            kvi_feature = None
        if "bandwidth" in self.tuned_history["Kvi"]:
            kvi_bandwidth = self.tuned_history["Kvi"]["bandwidth"]
            bandwidth+=(kvi_bandwidth)
        if "gain" in self.tuned_history["Kvi"]:
            kvi = self.tuned_history["Kvi"]["gain"]

        if "phase shift" in self.tuned_history["Kvp"]:
            kvp_feature = self.tuned_history["Kvp"]["phase shift"]
        elif "magnitude deviation" in self.tuned_history["Kvp"]:
            kvp_feature = self.tuned_history["Kvp"]["magnitude deviation"]
        else:
            kvp_feature = None
        if "bandwidth" in self.tuned_history["Kvp"]:
            kvp_bandwidth = self.tuned_history["Kvp"]["bandwidth"]
            bandwidth += kvp_bandwidth
        if "gain" in self.tuned_history["Kvp"]:
            kvp = self.tuned_history["Kvp"]["gain"]

        
        if "phase shift" in self.tuned_history["Kpi"]:
            kpi_feature = self.tuned_history["Kpi"]["phase shift"]
        elif "magnitude deviation" in self.tuned_history["Kpi"]:
            kpi_feature = self.tuned_history["Kpi"]["magnitude deviation"]
        else:
            kpi_feature = None
        if "bandwidth" in self.tuned_history["Kpi"]:
            kpi_bandwidth = self.tuned_history["Kpi"]["bandwidth"]
            bandwidth += kpi_bandwidth
        if "gain" in self.tuned_history["Kpi"]:
            kpi = self.tuned_history["Kpi"]["gain"]

        if "phase shift" in self.tuned_history["Kpp"]:
            kpp_feature = self.tuned_history["Kpp"]["phase shift"]
        elif "magnitude deviation" in self.tuned_history["Kpp"]:
            kpp_feature = self.tuned_history["Kpp"]["magnitude deviation"]
        else:
            kpp_feature = None
        if "bandwidth" in self.tuned_history["Kpp"]:
            kpp_bandwidth = self.tuned_history["Kpp"]["bandwidth"]
            bandwidth+=kpp_bandwidth
        if "gain" in self.tuned_history["Kpp"]:
            kpp = self.tuned_history["Kpp"]["gain"]

         # Create subplots
        fig, axs = plt.subplots(9, 1, figsize=(10, 20))
        idx = 0
        # Plot gains
        if kvi is not None:
            axs[idx].plot(kvi, marker='o', label='Kvi gain')
            axs[idx].set_title('Kvi Gain')
            axs[idx].set_xlabel('Iteration')
            axs[idx].set_ylabel('Gain')
            axs[idx].legend()
            idx = idx + 1
        
        if kvp is not None:
            axs[idx].plot(kvp, marker='o', label='Kvp gain')
            axs[idx].set_title('Kvp Gain')
            axs[idx].set_xlabel('Iteration')
            axs[idx].set_ylabel('Gain')
            axs[idx].legend()
            idx = idx + 1
            
        if kpi is not None:
            axs[idx].plot(kpi, marker='o', label='Kpi gain')
            axs[idx].set_title('Kpi Gain')
            axs[idx].set_ylabel('Gain')
            axs[idx].legend()
            idx = idx + 1

        if kpp is not None:
            axs[idx].plot(kpp, marker='o', label='Kpp gain')
            axs[idx].set_title('Kpp Gain')
            axs[idx].set_ylabel('Gain')
            axs[idx].legend()
            idx = idx + 1

        # Plot bandwidth
        axs[idx].plot(bandwidth, marker='o', label='Bandwidth')
        axs[idx].set_title('Bandwidth')
        axs[idx].set_ylabel('Bandwidth')
        axs[idx].legend()
        idx = idx + 1

        # Plot features (if they exist)
        if kpp_feature is not None:
            axs[idx].plot(kpp_feature, marker='o', label='Kpp Feature')
            axs[idx].legend()
            idx = idx + 1
        if kpi_feature is not None:
            axs[idx].plot(kpi_feature, marker='x', label='Kpi Feature')
            axs[idx].legend()
            idx = idx + 1
        if kvp_feature is not None:
            axs[idx].plot(kvp_feature, marker='s', label='Kvp Feature')
            axs[idx].legend()
            idx = idx + 1
        if kvi_feature is not None:
            axs[idx].plot(kvi_feature, marker='^', label='Kvi Feature')
            axs[idx].legend()
            idx = idx + 1
        
        # axs[idx].set_title('Features')
        # axs[idx].set_ylabel('Feature Value')
        
        

        # Adjust layout for better readability
        plt.tight_layout()
        

        plt.plot(bandwidth)
        plt.xlabel('iterations')
        plt.ylabel('bandwidth')
        plt.title('Tune History - Bandwidth')
        plt.grid(True)

        # Show the plots
        plt.show()

def main():
    interpolation = Intp(iter=12)
    # interpolation.inference()
    interpolation.lag_joint = 4

    interpolation.tune_gain(ServoGain.Velocity.value.ki, 0.1, 200, FeatureType.phase_shift)
    interpolation.reset_RT605()

    interpolation.tune_gain(ServoGain.Velocity.value.kp, 5, 100, FeatureType.phase_shift)
    interpolation.reset_RT605()

    # interpolation.tune_gain(ServoGain.Position.value.ki, 50, 150, FeatureType.magnitude_deviation)
    # interpolation.reset_RT605()

    interpolation.tune_gain(ServoGain.Position.value.kp, 5, 150, FeatureType.magnitude_deviation)
    interpolation.save_json_file()
    interpolation.reset_RT605()

    interpolation.plot_tune_history()

    # for iter in range(0, 1):
    
    #     interpolation.inference()
        
    #     interpolation.run(ServoGain.Velocity.value.kp, 0.01, 100)


    #     interpolation.run(ServoGain.Velocity.value.ki, 0.001, 50)

    #     interpolation.run(ServoGain.Position.value.kp, 0.01, 100)

    #     # interpolation.run(ServoGain.Position.value.ki, 0.001, 50)

    #     interpolation.plot_result()
    
    # interpolation.save_file()

def tune_loop_test(kp_min, kp_max, ki_min, ki_max, feature_type: FeatureType):
    interpolation = Intp(iter=10, pointsCount=1000)
    interpolation.inference()
    

    # interpolation.tune_loop(Loop_Type.vel, kp_min, kp_max, ki_min, ki_max, FeatureType.magnitude_deviation)

    # interpolation.reset_RT605()

    interpolation.tune_loop(Loop_Type.pos, kp_min, kp_max, ki_min, ki_max, FeatureType.magnitude_deviation)
    

    #interpolation.plot3D()

    # interpolation.save_json_file()

    interpolation.reset_RT605()


def test(iteration=6):
    
    interpolation = Intp(iter=10)
    tune_history = {}
    
    interpolation.rt605.run_HRSS_intp()
    interpolation.rt605.plot_joint()
    fig1 = interpolation.rt605.freq_response()
    interpolation.rt605.plot_error()
    tracking_err_before = copy.deepcopy(interpolation.rt605.q_pos_err)
    iterations = [8,11,8,8,8,8] 
    t = np.array(range(0,interpolation.rt605.arr_size))*interpolation.rt605.ts
    fig,ax = plt.subplots(6,1)

    # Set the same scale for each axis
    max_range = np.array([tracking_err_before[:,0].max()-tracking_err_before[:,0].min(), 
                        tracking_err_before[:,1].max()-tracking_err_before[:,1].min(),
                        tracking_err_before[:,2].max()-tracking_err_before[:,2].min(),
                        tracking_err_before[:,3].max()-tracking_err_before[:,3].min(),
                        tracking_err_before[:,4].max()-tracking_err_before[:,4].min(),
                        tracking_err_before[:,5].max()-tracking_err_before[:,5].min()]).max() / 2.0
    mid_q1_err = (tracking_err_before[:,0].max()+tracking_err_before[:,0].min()) * 0.5 
    mid_q2_err = (tracking_err_before[:,1].max()+tracking_err_before[:,1].min()) * 0.5 
    mid_q3_err = (tracking_err_before[:,2].max()+tracking_err_before[:,2].min()) * 0.5
    mid_q4_err = (tracking_err_before[:,3].max()+tracking_err_before[:,3].min()) * 0.5 
    mid_q5_err = (tracking_err_before[:,4].max()+tracking_err_before[:,4].min()) * 0.5 
    mid_q6_err = (tracking_err_before[:,5].max()+tracking_err_before[:,5].min()) * 0.5

    mod_q_err = (mid_q1_err,mid_q2_err,mid_q3_err,mid_q4_err,mid_q5_err,mid_q6_err)
    
    # for i in range(6):
    #     ax[i//2,i%2].set_title(f"joint{i+1}")
    #     ax[i//2,i%2].plot(t,self.q_pos_err[:,i])
    #     ax[i//2,i%2].grid(True)
    #     ax[i//2,i%2].set_ylim(mod_q_err[i] - 1.1 * max_range, mod_q_err[i]  + 1.1 * max_range)
    #     ax[i//2,i%2].set_xlabel("time(s)")
    #     ax[i//2,i%2].set_ylabel(r"$\theta$(deg)")    
    for i in range(6):
        # ax[i].set_title(f"joint{i+1}")
        ax[i].plot(t,tracking_err_before[:,i], label='original')
        # ax[i].plot(t,tracking_err_after[:,i], label='tuned')
        ax[i].grid(True)
        ax[i].set_ylim(mod_q_err[i] - 1.1 * max_range, mod_q_err[i]  + 1.1 * max_range)
        ax[5].set_xlabel("time(s)")
        ax[i].set_ylabel(r"$\theta$(deg)") 



    plt.suptitle('Joint angle error')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.1)
    plt.show()
    for iter in range(6):
        interpolation.inference()
        interpolation.iter = iterations[iter]

        interpolation.tune_gain(ServoGain.Velocity.value.ki, 0.1, 200, FeatureType.magnitude_deviation)
        interpolation.reset_RT605()

        interpolation.tune_gain(ServoGain.Velocity.value.kp, 5, 100, FeatureType.magnitude_deviation)
        interpolation.reset_RT605()

        interpolation.tune_gain(ServoGain.Position.value.kp, 5, 150, FeatureType.phase_shift)
        interpolation.reset_RT605()

        tune_history[f"iter{iter}"] = {}
        tune_history[f"iter{iter}"]["tune history"]=copy.deepcopy(interpolation.tuned_history)
        tune_history[f"iter{iter}"]["iter"] = iter
        tune_history[f"iter{iter}"]["lag joint"]=interpolation.lag_joint
        tune_history[f"iter{iter}"]["Bandwidth"]=interpolation.rt605.bandwidth
        print(tune_history)
        interpolation.tuned_history.clear()
        interpolation.tuned_history["iterations"] = {}
        interpolation.tuned_history["Kpp"] = {}
        interpolation.tuned_history["Kpi"] = {}
        interpolation.tuned_history["Kvp"] = {}
        interpolation.tuned_history["Kvi"] = {}
    tracking_err_after = copy.deepcopy(interpolation.rt605.q_pos_err)
    interpolation.rt605.run_HRSS_intp()
    interpolation.rt605.plot_joint()
    interpolation.rt605.freq_response()
    interpolation.rt605.plot_error()
    interpolation.save_json_file(tune_history)

    t = np.array(range(0,interpolation.rt605.arr_size))*interpolation.rt605.ts
    fig,ax = plt.subplots(6,1)

    # Set the same scale for each axis
    max_range = np.array([tracking_err_before[:,0].max()-tracking_err_before[:,0].min(), 
                        tracking_err_before[:,1].max()-tracking_err_before[:,1].min(),
                        tracking_err_before[:,2].max()-tracking_err_before[:,2].min(),
                        tracking_err_before[:,3].max()-tracking_err_before[:,3].min(),
                        tracking_err_before[:,4].max()-tracking_err_before[:,4].min(),
                        tracking_err_before[:,5].max()-tracking_err_before[:,5].min()]).max() / 2.0
    mid_q1_err = (tracking_err_before[:,0].max()+tracking_err_before[:,0].min()) * 0.5 
    mid_q2_err = (tracking_err_before[:,1].max()+tracking_err_before[:,1].min()) * 0.5 
    mid_q3_err = (tracking_err_before[:,2].max()+tracking_err_before[:,2].min()) * 0.5
    mid_q4_err = (tracking_err_before[:,3].max()+tracking_err_before[:,3].min()) * 0.5 
    mid_q5_err = (tracking_err_before[:,4].max()+tracking_err_before[:,4].min()) * 0.5 
    mid_q6_err = (tracking_err_before[:,5].max()+tracking_err_before[:,5].min()) * 0.5

    mod_q_err = (mid_q1_err,mid_q2_err,mid_q3_err,mid_q4_err,mid_q5_err,mid_q6_err)
    
    # for i in range(6):
    #     ax[i//2,i%2].set_title(f"joint{i+1}")
    #     ax[i//2,i%2].plot(t,self.q_pos_err[:,i])
    #     ax[i//2,i%2].grid(True)
    #     ax[i//2,i%2].set_ylim(mod_q_err[i] - 1.1 * max_range, mod_q_err[i]  + 1.1 * max_range)
    #     ax[i//2,i%2].set_xlabel("time(s)")
    #     ax[i//2,i%2].set_ylabel(r"$\theta$(deg)")    
    for i in range(6):
        # ax[i].set_title(f"joint{i+1}")
        ax[i].plot(t,tracking_err_before[:,i], label='original')
        ax[i].plot(t,tracking_err_after[:,i], label='tuned')
        ax[i].grid(True)
        ax[i].set_ylim(mod_q_err[i] - 1.1 * max_range, mod_q_err[i]  + 1.1 * max_range)
        ax[5].set_xlabel("time(s)")
        ax[i].set_ylabel(r"$\theta$(deg)") 



    plt.suptitle('Joint angle error')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.1)
    plt.show()



if __name__ == "__main__":
    # test(ServoGain.Position.value.kp, 1, 100, FeatureType.magnitude_deviation)
    # test(ServoGain.Position.value.kp, 1, 100, FeatureType.phase_shift)
    # test(ServoGain.Position.value.ki, 0.05, 10, FeatureType.magnitude_deviation)
    # test(ServoGain.Position.value.ki, 0.05, 10, FeatureType.phase_shift)
    # test(ServoGain.Velocity.value.kp, 5, 100, FeatureType.magnitude_deviation)
    # test(ServoGain.Velocity.value.kp, 5, 100, FeatureType.phase_shift)
    # test(ServoGain.Velocity.value.ki, 0.08, 10, FeatureType.magnitude_deviation)
    # test(ServoGain.Velocity.value.ki, 0.08, 10, FeatureType.phase_shift)


    # tune_loop_test(1,100,50,150,FeatureType.magnitude_deviation)


    test()