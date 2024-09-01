#%%
import os
from typing import Any
from libs.RobotManipulators import RT605_710
import libs.ServoDriver as dev
import numpy as np
import math
from numpy import linalg as LA

from threading import Thread
from libs.ServoMotor import ServoMotor
from libs.type_define import *
from libs import ControlSystem as cs
from libs import ServoDriver
from libs.ForwardKinematic import FowardKinematic
from libs.rt605_Gtorq_model import RT605_GTorq_Model
from libs.rt605_Friction_model import RT605_Friction_Model

import json
import time
import threading
import csv
from scipy.fft import fft, fftfreq, ifft
from scipy.signal import freqz
import statistics

### plot library
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.basemap import Basemap
from tqdm import tqdm
from tqdm.contrib import tzip


### Data base ###
import json



class RT605():
    def __init__(self,ts=0.0005) -> None:
        self.data = None
        self.ts = ts
        self.time = None


        self.q_c = None

        self.q1_c = None
        self.q2_c = None
        self.q3_c = None
        self.q4_c = None
        self.q5_c = None
        self.q6_c = None

        self.q_pos_err = None
        self.torque = None 
        self.q = None
        self.dq = None
        self.ddq = None


        self.x = None
        self.y = None
        self.z = None 
        self.x_c = None
        self.y_c = None
        self.z_c = None
        self.pitch = None
        self.roll = None
        self.yaw = None

        self.unstable_state = False

        self.contour_err = []
        self.circular_err = []
        self.ori_contour_err = []
        self.phase = []
        self.phase_delay = []
        self.tracking_err = []
        self.tracking_err_x = []
        self.tracking_err_y = []
        self.tracking_err_z = []
        self.tracking_err_pitch = []
        self.tracking_err_roll = []
        self.tracking_err_yaw = []

        self.bandwidth = []

        self.arr_size = None 

        self.path_mode = None # determine if it is XY(0)/YZ(1) circle or line(2)

        self.joints:ServoDriver.JointServoDrive = [None]*6 # Empty list to hold six joint instance

        self.forward_kinematic = FowardKinematic(unit='degree')

        self.compute_GTorque = RT605_GTorq_Model()
        self.compute_friction = RT605_Friction_Model()
        # self.motor_freq_response = Freq_Response()
 
        self.model_path = os.path.dirname(os.path.abspath(__file__)) + '/data/servos/'
        self.log_path = os.path.dirname(os.path.abspath(__file__)) + '/run/'
        #self.path_file_dir = './data/Path/'
        #self.path_name = 'XY_circle_path.txt'

        # circular test
        self.x_center = 0
        self.y_center = 0
        self.z_center = 0

        self.initialize_model()
        
    def setPosition(self, q:list):
        '''
        q: 【degree】
        '''
        for i in range(6):
            self.joints[i].setInitial(pos_init=q[i])

        # Servo joint and motor initialization
        self.initialize_model()
        

    def load_HRSS_trajectory(self,path_dir:str,prolong:bool=True):
        try:
            # self.data = np.genfromtxt(self.path_file_dir+self.path_name, delimiter=',')
            self.data = np.genfromtxt(path_dir, delimiter=',')
        except:
            return None

        # deterning if it is XY circular test or YZ circular test or line test
        if path_dir.find("XY") !=-1:
            self.path_mode = 0
        elif path_dir.find("YZ") != -1:
            self.path_mode = 1
        elif path_dir.find("line") != -1:
            self.path_mode = 2
        elif path_dir.find("sine") !=-1:
            self.path_mode = 3
        
        expand = int(len(self.data[:,0])*1.2)
        # cartesian command(mm)
        self.x_c = np.concatenate([self.data[:,0]/1000000, np.full((expand - len(self.data)), self.data[-1,0]/1000000)])
        self.y_c = np.concatenate([self.data[:,1]/1000000, np.full((expand - len(self.data)), self.data[-1,1]/1000000)])
        self.z_c = np.concatenate([self.data[:,2]/1000000, np.full((expand - len(self.data)), self.data[-1,2]/1000000)])

        # cartesian command(degree)
        self.pitch_c = np.concatenate([self.data[:, 3]/1000, np.full((expand - len(self.data)), self.data[-1,3]/1000)])  # A --> Ry
        self.roll_c = np.concatenate([self.data[:, 4]/1000, np.full((expand - len(self.data)), self.data[-1,4]/1000)])   # B --> -Rx  the coordinate of tool frame is up-side down
        self.yaw_c = np.concatenate([self.data[:,5]/1000, np.full((expand - len(self.data)), self.data[-1,5]/1000)])     # C --> Rz

        # joint command(degree)
        self.q1_c = np.concatenate([self.data[:,6]/1000, np.full((expand - len(self.data)), self.data[-1,6]/1000)])
        self.q2_c = np.concatenate([self.data[:,7]/1000, np.full((expand - len(self.data)), self.data[-1,7]/1000)])
        self.q3_c = np.concatenate([self.data[:,8]/1000, np.full((expand - len(self.data)), self.data[-1,8]/1000)])
        self.q4_c = np.concatenate([self.data[:,9]/1000, np.full((expand - len(self.data)), self.data[-1,9]/1000)])
        self.q5_c = np.concatenate([self.data[:,10]/1000, np.full((expand - len(self.data)), self.data[-1,10]/1000)])
        self.q6_c = np.concatenate([self.data[:,11]/1000, np.full((expand - len(self.data)), self.data[-1,11]/1000)])

        # Concatenate the arrays into a single 2-dimensional array
        self.q_c = np.column_stack((self.q1_c,self.q2_c,self.q3_c,
                                    self.q4_c,self.q5_c,self.q6_c))
        
        # print(f"q_c shape = {self.q_c.shape}")
        # set Initial condition of rt605
        for i in range(6):
            self.joints[i].setInitial(pos_init=self.q_c[0,i])
            # self.joints[i].motor.setInit(self.q_c[0,i]) 
        
        self.arr_size = self.q1_c.shape[0]

        # Sampling time
        self.time = self.ts * np.arange(0,self.arr_size)

        self.q_pos_err = np.zeros((self.arr_size,6))
        self.torque = np.zeros((self.arr_size,6))
        self.q = np.zeros((self.arr_size, 6))
        self.dq = np.zeros((self.arr_size, 6))
        self.ddq = np.zeros((self.arr_size, 6))

        self.x = np.zeros(self.arr_size)
        self.y = np.zeros(self.arr_size)
        self.z = np.zeros(self.arr_size)

        self.pitch = np.zeros(self.arr_size)
        self.roll = np.zeros(self.arr_size)
        self.yaw = np.zeros(self.arr_size)

        self.x_center = (min(self.x_c) + max(self.x_c))/2
        self.y_center = (min(self.y_c) + max(self.y_c))/2
        self.z_center = (min(self.z_c) + max(self.z_c))/2
        

        return self.q_c


    def setPID(self, id, gain:ServoGain, value:np.float32)->None:
        # if gain == "kvp":
        #     self.joints[id].setPID(ServoGain.Velocity.value.kp, value)
        # elif gain =="kvi":
        #     self.joints[id].setPID(ServoGain.Velocity.value.ki, value)
        # elif gain == "kpp":
        #     self.joints[id].setPID(ServoGain.Position.value.kp, value)
        # elif gain == "kpi":
        #     self.joints[id].setPID(ServoGain.Position.value.ki, value)
        # else:
        #     print("input argument error!!")
        self.joints[id].setPID(gain, value)

    def setMotorModel(self, id, component=str(), value=np.float32):
        if component == "Jm":
            self.joints[id].setMotorModel(MotorModel.Jm, value)
        elif component == "fric_vis":
            self.joints[id].setMotorModel(MotorModel.fric_vis, value)
        elif component == "fric_c":
            self.joints[id].setMotorModel(MotorModel.fric_c, value)
        elif component == "fric_dv":
            self.joints[id].setMotorModel(MotorModel.fric_dv, value)
        else:
            print("input argument error!!")

    def initialize_model(self, servo_file_dir:str=None):
        # print(servo_file_dir)
        # for loop run six joint initialization
        for i in range(6):
            model_path_name = f"j{i+1}/j{i+1}.sdrv"
            if servo_file_dir==None:
                self.joints[i] = ServoDriver.JointServoDrive(id=i,saved_model=self.model_path+model_path_name)
            else:
                self.joints[i] = ServoDriver.JointServoDrive(id=i,saved_model=servo_file_dir+model_path_name)
    
    def resetServoModel(self):
        """
        reset servo model(e.g: Jm, bm...) to its initial values
        """
        # This function is for reset the servo model parameter
        for i,joint in self.joints:
            model_path_name = f"j{i+1}/j{i+1}.sdrv"
            joint.ImportServoModel(saved_model=self.model_path+model_path_name)

    def resetServoDrive(self)->None:
        """
        reset each joints to its initial position
        """
        for i, joint in enumerate(self.joints):
            joint.setInitial(self.q_c[0, i])

    def resetPID(self):
        """
        reset Servo gain to its initial value
        """
        # This function is for reseting the servo gain
        self.initialize_model()

    def __call__(self,q_ref:np.ndarray)->tuple[float, float, float, float, float, float]:
        """
        Processes the input reference joint positions and returns the computed
        position and orientation of the end effector.
        
        Parameters:
        q_ref (np.ndarray): A numpy array of shape (6,) representing the 
                            reference joint positions.
        
        Returns:
        tuple: A tuple containing the x, y, z coordinates and the pitch, roll, 
               yaw angles of the end effector.
        """
        if not isinstance(q_ref,np.ndarray):
            raise TypeError("Input datamust be a Numpy array with size")
        if q_ref.shape != (6,):
            raise ValueError("Input data must have shape (6,)")
        
        g_tor = np.zeros(6,dtype=np.float32)
        fric_tor = np.zeros(6,dtype=np.float32)
        # q = np.zeros(6,dtype=np.float32)
        # dq = np.zeros(6,dtype=np.float32)
        # ddq = np.zeros(6,dtype=np.float32)

                
        for idx in range(6):
            pos,vel,acc,tor,pos_err, vel_cmd  = self.joints[idx](q_ref[idx],g_tor[idx])
            self.q[idx] = pos 
            self.dq[idx] = vel 
            self.ddq[idx] = acc

        g_tor = self.compute_GTorque(self.q[1],self.q[2],self.q[3],self.q[4],self.q[5])
            
        fric_tor = self.compute_friction(self.q[0],self.q[1],self.q[2],
                                        self.q[3],self.q[4],self.q[5]) #TODO

        x,y,z,pitch,roll,yaw = self.forward_kinematic(
                                (self.q[0],self.q[1],self.q[2],self.q[3],self.q[4],self.q[5]))
        
        
        return x,y,z,pitch,roll,yaw

    def run_HRSS_intp(self)->None:
        """
        Executes the main control loop, updating joint positions, velocities,
        accelerations, torques, and calculating various error metrics for the
        robotic system.
        """
        g_tor = np.zeros(6,dtype=np.float32)
        fric_tor = np.zeros(6,dtype=np.float32)

        self.contour_err = []
        self.circular_err = []
        self.ori_contour_err = []
        self.phase = []
        self.phase_delay = []
        self.tracking_err_x = []
        self.tracking_err_y = []
        self.tracking_err_z = []
        self.tracking_err_pitch = []
        self.tracking_err_roll = []
        self.tracking_err_yaw = []


        for i, q_ref in enumerate(tzip(self.q1_c,self.q2_c,self.q3_c,self.q4_c,self.q5_c,self.q6_c)):
            
            for idx in range(6):
                pos,vel,acc,tor,pos_err, _ = self.joints[idx](q_ref[idx],g_tor[idx])
                self.q[i][idx] = pos 
                self.dq[i][idx] = vel 
                self.ddq[i][idx] = acc 
                self.q_pos_err[i][idx] = pos_err
                self.torque[i][idx] = tor

            g_tor = self.compute_GTorque(self.q[i][1],self.q[i][2],self.q[i][3],
                                            self.q[i][4],self.q[i][5])
            
            fric_tor = self.compute_friction(self.q[i][0],self.q[i][1],self.q[i][2],
                                            self.q[i][3],self.q[i][4],self.q[i][5]) #TODO

            self.x[i],self.y[i],self.z[i],self.pitch[i],self.roll[i],self.yaw[i] = self.forward_kinematic(
                                    (self.q[i,0],self.q[i,1],self.q[i,2],
                                        self.q[i,3],self.q[i,4],self.q[i,5]))
            
            self.tracking_err_x.append(self.x[i]-self.x_c[i])
            self.tracking_err_y.append(self.y[i]-self.y_c[i])
            self.tracking_err_z.append(self.z[i]-self.z_c[i])
            self.tracking_err_pitch.append((self.pitch[i]-self.pitch_c[i]+180)%360-180)
            self.tracking_err_roll.append((self.roll[i]-self.roll_c[i]+180)%360-180)
            self.tracking_err_yaw.append((self.yaw[i]-self.yaw_c[i]+180)%360-180)

            
            self.tracking_err.append(LA.norm([self.tracking_err_x, self.tracking_err_y, self.tracking_err_z]))

            
            self.contour_err.append(self.computeCountourErr(self.x[i],self.y[i],self.z[i]))
            self.ori_contour_err.append(self.compute_ori_contour_error(self.pitch[i],self.roll[i],self.yaw[i]))
            self.circular_err.append(self.computeCircularErr(self.x[i], self.y[i]))


            current_angle = self.compute_angle(self.x[i],self.y[i],self.z[i])
            ref_angle = self.compute_angle(self.x_c[i], self.y_c[i], self.z_c[i])

            self.phase.append(current_angle)
            delay = ref_angle - current_angle
            if(delay<-np.pi):
                delay+=2*np.pi
            if(delay>np.pi):
                delay-=2*np.pi
            self.phase_delay.append(delay)

        self.resetServoDrive()
        for joint in self.joints:
            if joint.unstable_state[0] == True:
                print("Servo unstatble")
                self.unstable_state = True

    def save_log(self,save_dir=None):
        self.log_path = save_dir + '/log/'

        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)

        ## System log ###
        np.savetxt(self.log_path+'joint_pos_error.txt',self.q_pos_err,delimiter=',',header='Joint1, Joint2, Joint3, Joint4, Joint5, Joint6', fmt='%10f')
        np.savetxt(self.log_path+'joint_pos.txt',self.q,delimiter=',',header='Joint1, Joint2, Joint3, Joint4, Joint5, Joint6', fmt='%10f')
        np.savetxt(self.log_path+'tor.txt',self.torque,delimiter=',',header='Joint1, Joint2, Joint3, Joint4, Joint5, Joint6', fmt='%10f')
        np.savetxt(self.log_path+'contour_error.txt', self.contour_err)
        np.savetxt(self.log_path+'tracking_error.txt',self.tracking_err, 'fmt=%10f')
        np.savetxt(self.log_path+'tracking_error_x.txt',self.tracking_err_x, 'fmt=%10f')
        np.savetxt(self.log_path+'tracking_error_y.txt',self.tracking_err_y, 'fmt=%10f')
        np.savetxt(self.log_path+'tracking_error_z.txt',self.tracking_err_z, 'fmt=%10f')
        np.savetxt(self.log_path+'tracking_error_pitch.txt',self.tracking_err_pitch, 'fmt=%10f')
        np.savetxt(self.log_path+'tracking_error_roll.txt',self.tracking_err_roll, 'fmt=%10f')
        np.savetxt(self.log_path+'tracking_error_yaw.txt',self.tracking_err_yaw, 'fmt=%10f')

        np.savetxt(self.log_path+'joint_vel.txt', self.dq,delimiter=',',header='Joint1, Joint2, Joint3, Joint4, Joint5, Joint6', fmt='%10f')
        np.savetxt(self.log_path+'joint_acc.txt', self.ddq,delimiter=',',header='Joint1, Joint2, Joint3, Joint4, Joint5, Joint6', fmt='%10f')    

    def plot_joint(self, show=True):

        ### plot the result ###
        t = np.array(range(0,self.arr_size))*self.ts
        fig,ax = plt.subplots(3,2, figsize=(5.4,4.5))
        
        # Set the same scale for each axis
        max_range = np.array([self.q[:,0].max()-self.q[:,0].min(), 
                            self.q[:,1].max()-self.q[:,1].min(),
                            self.q[:,2].max()-self.q[:,2].min(),
                            self.q[:,3].max()-self.q[:,3].min(),
                            self.q[:,4].max()-self.q[:,4].min(),
                            self.q[:,5].max()-self.q[:,5].min()]).max() / 2.0
        mid_q1 = (self.q[:,0].max()+self.q[:,0].min()) * 0.5 
        mid_q2 = (self.q[:,1].max()+self.q[:,1].min()) * 0.5 
        mid_q3 = (self.q[:,2].max()+self.q[:,2].min()) * 0.5
        mid_q4 = (self.q[:,3].max()+self.q[:,3].min()) * 0.5 
        mid_q5 = (self.q[:,4].max()+self.q[:,4].min()) * 0.5 
        mid_q6 = (self.q[:,5].max()+self.q[:,5].min()) * 0.5

        mid_q = (mid_q1,mid_q2,mid_q3,mid_q4,mid_q5,mid_q6)

        for i in range(6):
            ax[i//2,i%2].set_title(f"joint{i+1}")
            ax[i//2,i%2].plot(t,self.q[:,i],label='actual')
            ax[i//2,i%2].plot(t,self.q_c[:,i],label='ref')
            ax[i//2,i%2].grid(True)
            ax[i//2,i%2].set_ylim(mid_q[i] - 1.1 * max_range, mid_q[i] + 1.1 * max_range)
            ax[i//2,i%2].set_xlabel("time(s)")
            ax[i//2,i%2].set_ylabel(r"$\theta$(deg)")
        ax[0,0].legend(loc='best')

        plt.suptitle('Joint angle')
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    def plot_error(self, show=True):
        t = np.array(range(0,self.arr_size))*self.ts
        fig,ax = plt.subplots(6,1)

        # Set the same scale for each axis
        max_range = np.array([self.q_pos_err[:,0].max()-self.q_pos_err[:,0].min(), 
                            self.q_pos_err[:,1].max()-self.q_pos_err[:,1].min(),
                            self.q_pos_err[:,2].max()-self.q_pos_err[:,2].min(),
                            self.q_pos_err[:,3].max()-self.q_pos_err[:,3].min(),
                            self.q_pos_err[:,4].max()-self.q_pos_err[:,4].min(),
                            self.q_pos_err[:,5].max()-self.q_pos_err[:,5].min()]).max() / 2.0
        mid_q1_err = (self.q_pos_err[:,0].max()+self.q_pos_err[:,0].min()) * 0.5 
        mid_q2_err = (self.q_pos_err[:,1].max()+self.q_pos_err[:,1].min()) * 0.5 
        mid_q3_err = (self.q_pos_err[:,2].max()+self.q_pos_err[:,2].min()) * 0.5
        mid_q4_err = (self.q_pos_err[:,3].max()+self.q_pos_err[:,3].min()) * 0.5 
        mid_q5_err = (self.q_pos_err[:,4].max()+self.q_pos_err[:,4].min()) * 0.5 
        mid_q6_err = (self.q_pos_err[:,5].max()+self.q_pos_err[:,5].min()) * 0.5

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
            ax[i].plot(t,self.q_pos_err[:,i])
            ax[i].grid(True)
            ax[i].set_ylim(mod_q_err[i] - 1.1 * max_range, mod_q_err[i]  + 1.1 * max_range)
            ax[5].set_xlabel("time(s)")
            ax[i].set_ylabel(r"$\theta$(deg)") 



        plt.suptitle('Joint angle error')
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.1)

        if show:
            plt.show()
        return fig

    def plot_cartesian(self, show=True):
        t = np.array(range(0,self.arr_size))*self.ts
        # Set the same scale for each axis
        max_range = np.array([self.x.max()-self.x.min(), self.y.max()-self.y.min(), self.z.max()-self.z.min()]).max() / 2.0 * 1000
        mid_x = (self.x.max()+self.x.min()) * 0.5 * 1000
        mid_y = (self.y.max()+self.y.min()) * 0.5 * 1000
        mid_z = (self.z.max()+self.z.min()) * 0.5 * 1000
        
        fig1,ax = plt.subplots(3,1)
        ax[0].plot(t,self.x*1000, label='actual')
        ax[0].plot(t,self.x_c*1000, label='ref')
        ax[0].set_xlabel('time(s)')
        ax[0].set_ylabel('X(mm)')
        ax[0].set_ylim(mid_x - 1.1 * max_range, mid_x + 1.1 * max_range)
        ax[0].grid(True)
        ax[0].legend(loc='best')

        ax[1].plot(t,self.y*1000)
        ax[1].plot(t,self.y_c*1000)
        ax[1].set_xlabel('time(s)')
        ax[1].set_ylabel('Y(mm)')
        ax[1].set_ylim(mid_y - 1.1 * max_range, mid_y + 1.1 * max_range)
        ax[1].grid(True)

        ax[2].plot(t,self.z*1000)
        ax[2].plot(t,self.z_c*1000)
        ax[2].set_xlabel('time(s)')
        ax[2].set_ylabel('Z(mm)')
        ax[2].set_ylim(mid_z - 1.1 * max_range, mid_z  + 1.1 * max_range)
        ax[2].grid(True)


        # Create 3D plot
        fig2 = plt.figure(figsize=(3.6,3.6))
        ax = plt.axes(projection='3d')

        # Add data to plot
        # ax.scatter(x_c, y_c, z_c, s=1)
        ax.scatter(self.x*1000,self.y*1000,self.z*1000,s=1,label="actual")
        ax.scatter(self.x_c*1000, self.y_c*1000, self.z_c*1000, s=1,label="ref")
        
        ax.scatter(self.x_c[0]*1000,self.y_c[0]*1000,self.z_c[0]*1000, c='red',marker='*',s=100)
        ax.text(self.x_c[0]*000,self.y_c[0]*1000,self.z_c[0]*1000,"start",color='red')
        ax.legend(loc='best')

        # Set the same scale for each axis
        max_range = np.array([self.x.max()-self.x.min(), self.y.max()-self.y.min(), self.z.max()-self.z.min()]).max() / 2.0 * 1000
        mid_x = (self.x.max()+self.x.min()) * 0.5 * 1000
        mid_y = (self.y.max()+self.y.min()) * 0.5 * 1000
        mid_z = (self.z.max()+self.z.min()) * 0.5 * 1000
        ax.set_xlim(mid_x -1.1 *  max_range, mid_x +1.1 *  max_range)
        ax.set_ylim(mid_y -1.1 *  max_range, mid_y +1.1 *  max_range)
        ax.set_zlim(mid_z -1.1 *  max_range, mid_z +1.1 *  max_range)


        # Set labels and title  
        ax.set_xlabel('X(mm)')
        ax.set_ylabel('Y(mm)')
        ax.set_zlabel('Z(mm)')
        ax.set_title('3D XYZ plot')

        # Show the plot
        if show:
            plt.show()
        else:
            plt.close(fig1)
            plt.close(fig2)

        return fig1, fig2
    
    def plot_phase_delay(self, show=True):
        """
        compute/plot? the phase delay of circular test
        """
        if self.path_mode>1:
            return 
        
        phase = np.zeros(self.arr_size, dtype=np.float32)
        delay = np.zeros(self.arr_size, dtype=np.float32)

        x_start = self.x_c[0]
        y_start = self.y_c[0]
        
        for i, (x,y,z, x_c, y_c, z_c) in enumerate(zip(self.x,self.y,self.z, self.x_c, self.y_c, self.z_c)):


            v_original = (x_start - self.x_center, y_start - self.y_center)
            v = (x - self.x_center, y - self.y_center)
            v_c = (x_c - self.x_center, y_c - self.y_center)

            

            # Calculate magnitudes
            magnitude_v_original = math.sqrt((x_start - self.x_center)**2 + (y_start - self.y_center)**2)
            magnitude_v = math.sqrt((x - self.x_center)**2 + (y - self.y_center)**2)
            magnitude_v_c = math.sqrt((x_c - self.x_center)**2 + (y_c - self.y_center)**2)

            # Calculate dot product
            dot_product = np.dot(v_original, v) #v1[0] * v2[0] + v1[1] * v2[1]
            cosine_value = dot_product / (magnitude_v_original * magnitude_v)

            angle_radians_act = math.acos(np.clip(cosine_value, -1.0, 1.0))

            # Check the sign of the cross product to determine the quadrant
            cross_product = v_original[0] * v[1] - v_original[1] * v[0]
            if cross_product < 0:
                # Angle is in the range of 180 to 360 degrees
                angle_radians_act = 2*np.pi - angle_radians_act

            # Calculate dot product
            dot_product = np.dot(v_original, v_c) #v1[0] * v2[0] + v1[1] * v2[1]
            

            cosine_value = dot_product / (magnitude_v_original * magnitude_v_c)

            angle_radians_ref = math.acos(np.clip(cosine_value, -1.0, 1.0))

            # Check the sign of the cross product to determine the quadrant
            cross_product = v_original[0] * v_c[1] - v_original[1] * v_c[0]
            if cross_product < 0:
                # Angle is in the range of 180 to 360 degrees
                angle_radians_ref = 2*np.pi - angle_radians_ref

            

            phase[i] = angle_radians_act
            delay[i] = angle_radians_ref - angle_radians_act
            if(abs( -2*np.pi - delay[i])<0.2):
                delay[i]+=2*np.pi
            if(abs( 2*np.pi - delay[i]<0.2)):
               delay[i]-=2*np.pi
                

            # print(delay[i])

            

        fig = plt.figure(figsize=(6,6))
        ax = plt.subplot(111, polar=True)
        ax.plot(phase, delay)
        ax.grid(True)

        # plt.plot(delay)
    

        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig
    
    def plot_circular_polar_plot(self, show=True):
        if self.path_mode >1:
            print('path mode error')
            return 
        # Create circular error polar plot
        fig1 = plt.figure(figsize=(6,6))
        ax = plt.subplot(111, polar=True)
        ax.plot(self.phase, self.circular_err)
        ax.grid(True)

        # Create height error polar plot
        fig2 = plt.figure(figsize=(6,6))
        ax = plt.subplot(111, polar=True)
        ax.plot(self.phase, self.tracking_err_z)
        ax.grid(True)

        if show == False:
            plt.close(fig1)
            plt.close(fig2)
        else:
            plt.show()


        return fig1, fig2

    def plot_polar(self, show = True):
        """
        only works in circular mode
        return three figure object, contour_err, orientation_contou_err, phase delay
        """
        if self.path_mode > 1:
            print('path mode error')
            return
       
        # Create contour err polar plot
        fig1 = plt.figure(figsize=(6,6))
        ax = plt.subplot(111, polar=True)
        ax.plot(self.phase, self.contour_err)
        ax.grid(True)
        
        # Create orientation error polar plot
        fig2 = plt.figure(figsize=(6,6))
        ax = plt.subplot(111, polar=True)
        ax.plot(self.phase, self.ori_contour_err)
        ax.grid(True)

        # Create phase delay polar plot
        fig3 = plt.figure(figsize=(6,6))
        ax = plt.subplot(111, polar=True)
        ax.plot(self.phase, self.phase_delay)
        ax.grid(True)

        

        if show == False:
            plt.close(fig1)
            plt.close(fig2)
            plt.close(fig3)
        else:
            plt.show()


        return fig1, fig2, fig3
    
    def compute_angle(self, x,y,z):
        
        v1 = (self.x_c[0] - self.x_center, self.y_c[0] - self.y_center)
        v2 = (x - self.x_center, y - self.y_center)

        dot_product = v1[0]*v2[0] + v1[1]*v2[1]
        magnitude_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
        magnitude_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
        cos_theta = np.clip((dot_product / (magnitude_v1 * magnitude_v2)),-1.0, 1.0)
        
        theta = math.acos(cos_theta)

        if v1[0] * v2[1] - v1[1] * v2[0] < 0:
            # Angle is in the range of 180 to 360 degrees
            theta = 2*np.pi - theta

        return theta
    
    def computeCircularErr(self, x, y):

        if self.path_mode >1:
            # print('Error: wrong path Mode')
            return None
        
        R = math.sqrt((self.x_c[0] - self.x_center)**2 + (self.y_c[0] - self.y_center)**2)
        c_err = math.sqrt(((x-self.x_center)**2 + (y-self.y_center)**2))

        return c_err - R

    def computeCountourErr(self,x,y,z):
        #This function use the mathemetical formula to compute the circular trajectory contouring error
        #Which can reduce the compute complexity from O(n^2) to O(n)
        # print(x,y,z)

        c_err = 0

        # print(self.path_mode)
        if self.path_mode == 0: # XY circular test
            # print(x_center, y_center, z_center)
            # print(x,y,z)
            R = math.sqrt((self.x_c[0] - self.x_center)**2 + (self.y_c[0] - self.y_center)**2)
            c_err = math.sqrt((math.sqrt((x-self.x_center)**2 + (y-self.y_center)**2) - R)**2 + (z - self.z_center)**2)   
        elif self.path_mode == 1: # YZ circular test
            R = math.sqrt((self.y_c[0] - self.y_center)**2 + (self.z_c[0] - self.z_center)**2)
            c_err = math.sqrt((math.sqrt((y-self.y_center)**2 + (z-self.z_center)**2) - R)**2 + (x - self.x_center)**2)
        elif self.path_mode == 2: # line circular test
            if self.x_c[0] != self.x_c[100]: # X direction line
                c_err = math.sqrt((y - self.y_center)**2 + (z - self.z_center)**2)
            elif self.y_c[0] != self.y_c[100]: # Y direction line
                c_err = math.sqrt((x - self.x_center)**2 + (z - self.z_center)**2)
            elif self.z_c[0] != self.z_c[100]: # Z direction line
                c_err = math.sqrt((x - self.x_center)**2 + (y - self.y_center)**2)
        # print(f'{x,y,z}: {c_err}')

        return c_err
    
    def euler_to_unit_vector(self, pitch, roll, yaw):
        pitch = np.radians(pitch) # convert from degree to radian
        roll = np.radians(roll)
        yaw = np.radians(yaw)
        

        # Convert Euler angles to rotation matrix
        yaw_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw),  np.cos(yaw), 0],
                    [0,        0,        1]])

        pitch_matrix = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0,        1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])

        roll_matrix = np.array([[1, 0,        0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll),  np.cos(roll)]])
        
        # Combine rotations
        rotation_matrix = roll_matrix @ pitch_matrix @ yaw_matrix
        # Apply the combined rotation to a unit vector (1, 0, 0) representing the positive X-axis
        unit_vector = rotation_matrix @ [1, 0, 0]
        
        # Get the unit vector along the z-axis
        # Normalize to obtain a unit vector
        unit_vector = unit_vector / np.linalg.norm(unit_vector)
        return unit_vector
    
    def compute_ori_contour_error(self,pitch, roll, yaw):
        if self.path_mode == 3 or self.path_mode == 4:
            return 0
        # Convert Euler angles to unit vectors
        v1 = self.euler_to_unit_vector(pitch, roll, yaw)
        v2 = self.euler_to_unit_vector(self.pitch_c[0], self.roll_c[0], self.yaw_c[0]) 
        # print(v1,v2)
        # print(v1, v2)
        # Compute the dot product
        dot_product = np.dot(v1, v2)
        
        cross_product = v1[0]*v2[1] - v1[1]*v2[0]
       
        # Compute the angle between the vectors
        angle = np.arccos(dot_product)

        
        return np.degrees(angle)
    
    def computeTrackErr(self,x,y,z,pitch, roll, yaw):
        t_err_x = x - self.x_c
        t_err_y = y - self.y_c
        t_err_z = z - self.z_c
        t_err_pitch = pitch - self.pitch_c
        t_err_roll = roll - self.roll_c
        t_err_yaw = yaw - self.yaw_c

        t_err = math.sqrt(t_err_x**2 + t_err_y**2 + t_err_z**2)

        return t_err, t_err_x, t_err_y, t_err_z, t_err_pitch, t_err_roll, t_err_yaw
    
    def sweep(self,fs=2000, f0=0.1, f1=1000, t0=0, t1=1, a0=0.1, a1=0.001, tau=0.2, show=True):
        '''
        Determine the frequency response of the system:
        fs: sampling rate
        f0: start frequency
        f1: end frequency
        t0: start time
        t1: duration of chirp
        a0: start amplitude
        a1: end amplitude
        tau: time constant (in seconds) -- for exponential decay
        '''
        # print(fs)
        # t1 = self.arr_size*self.ts
        t = np.linspace(t0, t1, int(fs * (t1 - t0)), endpoint=False)

        # Generate chirp sine signal
        amp_decay = a0 * np.exp(-t / tau)
        f = f0 + (f1 - f0) / (t1 - t0) * t
        chirp_sine = amp_decay * np.sin(2 * np.pi * f * t)

        shape = (6, 6, len(t))

        q_c = np.zeros(shape)

        # Initialize q_c with initial joint position
        for i in range(6):
            for j in range(6):
                q_c[i][j] = np.full_like(t, self.q_c[0][j])
                if i == j:
                    q_c[i][j] += chirp_sine

        q = np.zeros(6)

        
        self.bandwidth = []

        fig, ax = plt.subplots(2, 1, figsize=(4.3, 4.1))

        # Determine frequency response of each joint
        for joint_num in range(6):
            output = []
            g_tor = np.zeros(6, dtype=np.float32)
            fric_tor = np.zeros(6, dtype=np.float32)
            for time_idx, time_val in enumerate(t):
                for idx in range(6):
                    q[idx], _, _, _, _, _ = self.joints[idx](q_c[joint_num][idx][time_idx], g_tor[idx])
                    if joint_num == idx:
                        output.append(q[idx])

                g_tor = self.compute_GTorque(*q[1:])
                fric_tor = self.compute_friction(*q)
            
            # plt.plot(t, output, label='Output')
            # plt.plot(t, q_c[joint_num][joint_num], label='q_c[joint_num][joint_num]')
            # plt.legend()
            # plt.show()
            yf = np.fft.fft(output) * 2/len(output)
            xf = np.fft.fft(q_c[joint_num][joint_num])* 2/len(output)
            # print(q_c[joint_num][joint_num].shape)
            # print(xf.shape)
            

            freqs = np.fft.fftfreq(len(yf)) * fs
            mag = np.abs(yf / xf)

            
            # Finding the index of the frequency where magnitude reaches -3 dB
            dB_threshold = -3
            diff = 20 * np.log10(mag[:len(mag) // 2]) - dB_threshold

            index = np.argmax(diff < 0)

            self.bandwidth.append(freqs[index])

            phase = np.angle(yf / xf)
            
            if show:
                
                ax[0].semilogx(freqs[:len(freqs) // 2], 20 * np.log10(mag[:len(mag) // 2]),
                                label=f"joint {joint_num + 1} - {freqs[index]:.2f} Hz")
                ax[0].set_xlabel('Frequency [Hz]')
                ax[0].set_ylabel('Magnitude [dB]')
                ax[0].grid(True)
                ax[0].set_xlim([f0, f1])
                ax[1].semilogx(freqs[:len(freqs) // 2], phase[:len(phase) // 2])
                ax[1].set_xlabel('Frequency [Hz]')
                ax[1].set_ylabel('Phase [rad]')
                ax[1].grid(True)
                ax[1].set_xlim([f0, f1])
                ax[0].legend()
        self.resetServoDrive()
        
        if show and fig is not None:
            plt.show()
        else:
            plt.close(fig)

        return fig
    
    def freq_response(self,mode='pos', show=True):
        
        self.bandwidth = []

        fig, ax = plt.subplots(2, 1, figsize=(8, 10))

        for i, joint in enumerate(self.joints):
            
            mag, phase, om = joint.freq_response(False, loop_mode=mode)

            # Interpolate omega and magnitude_response to num_points
            omega_interp = np.linspace(om[0], om[-1], 10000)
            magnitude_interp = np.interp(omega_interp, om, mag)
            phase_interp = np.interp(omega_interp, om, phase)

            # Convert magnitude response to dB
            magnitude_db = 20 * np.log10(magnitude_interp)

            max_magnitude_db = np.max(magnitude_db)
            index_bandwidth = np.argmin(abs(magnitude_db - (-3)))

            self.bandwidth.append(omega_interp[index_bandwidth]/ (2*np.pi))
            # print(omega_interp[index_bandwidth]/ (2*np.pi))

            # Plot magnitude response
            if show:
                
                ax[0].semilogx(omega_interp/ (2*np.pi), 20 * np.log10(magnitude_interp),
                                    label=f"joint {i + 1} - {omega_interp[index_bandwidth]/ (2*np.pi):.2f} Hz")
                ax[0].set_xlabel('Frequency (Hz)')
                ax[0].set_ylabel('Magnitude (dB)')
               
                ax[0].grid(True)
                ax[0].set_xlim([0.1, 100])
                ax[0].legend(loc='lower left', fontsize=14)
                ax[1].semilogx(omega_interp/ (2*np.pi), phase_interp)
                ax[1].set_xlabel('Frequency [Hz]')
                ax[1].set_ylabel('Phase [rad]')
                ax[1].grid(True)
                ax[1].set_xlim([0.1, 100])
                
        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_torq(self):

        plt.plot( )
        for i in range(6):
            plt.plot(self.time,self.torque[:,i], label=f'link {i+1}')
        plt.title('torque')
        plt.grid(True)
        plt.legend()
        plt.show()

    def pso_tune_gain_update(self):
        """
            this function is seted up temperary for tuning PID gain using PSO algorithm
        """
        g_tor = np.zeros(6,dtype=np.float32)
        fric_tor = np.zeros(6,dtype=np.float32)

        contour_err_sum = 0
        err = 0

        c_err = []
        ori_c_err = []

        for i, q_ref in enumerate(zip(self.q1_c,self.q2_c,self.q3_c,self.q4_c,self.q5_c,self.q6_c)):
            # print(q_ref)
            for idx in range(6):
                pos,vel,acc,tor,pos_err, vel_cmd = self.joints[idx](q_ref[idx],g_tor[idx])
                self.q[i][idx] = pos 
                self.dq[i][idx] = vel 
                self.ddq[i][idx] = acc 
                self.q_pos_err[i][idx] = pos_err
                self.torque[i][idx] = tor
            # print(self.q[i])

            g_tor = self.compute_GTorque(self.q[i][1],self.q[i][2],self.q[i][3],
                                            self.q[i][4],self.q[i][5])
            
            fric_tor = self.compute_friction(self.q[i][0],self.q[i][1],self.q[i][2],
                                            self.q[i][3],self.q[i][4],self.q[i][5]) #TODO

            self.x[i],self.y[i],self.z[i],self.pitch[i],self.roll[i],self.yaw[i] = self.forward_kinematic(
                                    (self.q[i,0],self.q[i,1],self.q[i,2],
                                        self.q[i,3],self.q[i,4],self.q[i,5]))

            c_err.append(self.computeCountourErr(self.x[i],self.y[i],self.z[i])) # tool path contour error
            contour_err_sum  = contour_err_sum + c_err[i]**2

            ori_c_err.append(self.compute_ori_contour_error(self.pitch[i],self.roll[i],self.yaw[i])) # orientation contour error
            

            # print( self.computeCountourErr(self.x[i],self.y[i],self.z[i]))
            # err = err + ((self.x_c[i]-self.x[i]) + (self.y_c[i]-self.y[i]) + (self.z_c[i]-self.z[i]))**2

        # loss = max(contour_error) + standard deviation(contour_err) + mean(contour_error)
        loss = max([abs(num) for num in c_err]) + statistics.mean(c_err) + statistics.stdev(c_err) + \
                max([abs(num) for num in ori_c_err]) + statistics.mean(ori_c_err) + statistics.stdev(ori_c_err)
        
        # print(contour_err_sum)
        
        return loss