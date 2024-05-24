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
from libs.bode_plot import Freq_Response

import json
import time
import threading
import csv
from scipy.fft import fft, fftfreq, ifft

### plot library
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.basemap import Basemap

### Data base ###
import pymysql
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

        self.contour_err = []
        self.tracking_err = []
        self.tracking_err_x = []
        self.tracking_err_y = []
        self.tracking_err_z = []
        self.tracking_err_pitch = []
        self.tracking_err_roll = []
        self.tracking_err_yaw = []

        self.bandwidth = []

        self.arr_size = None 


        self.thread_list = []

        self.path_mode = None # determine if it is XY(0)/YZ(1) circle or line(2)

        self.joints:ServoDriver.JointServoDrive = [None]*6 # Empty list to hold six joint instance

  
        self.forward_kinematic = FowardKinematic(unit='degree')

        self.compute_GTorque = RT605_GTorq_Model()
        self.compute_friction = RT605_Friction_Model()
        
        self.motor_freq_response = Freq_Response()

        self.model_path = './data/servos/'
        self.log_path = './run/'
        #self.path_file_dir = './data/Path/'
        #self.path_name = 'XY_circle_path.txt'
        
        self.progress = 0 #for fun 


        

    def load_HRSS_trajectory(self,path_dir:str):
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
        
        # cartesian command(mm)
        self.x_c = self.data[:,0]/1000000
        self.y_c = self.data[:,1]/1000000
        self.z_c = self.data[:,2]/1000000

        # cartesian command(degree)
        self.pitch_c = self.data[:, 3]/1000
        self.yaw_c = self.data[:, 4]/1000
        self.roll_c = self.data[:,5]/1000

        # joint command(degree)
        self.q1_c = self.data[:,6]/1000
        self.q2_c = self.data[:,7]/1000
        self.q3_c = self.data[:,8]/1000
        self.q4_c = self.data[:,9]/1000
        self.q5_c = self.data[:,10]/1000
        self.q6_c = self.data[:,11]/1000

        

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

        

        return self.q_c


    def setPID(self, id, gain=str(), value=np.float32):
        if gain == "kvp":
            self.joints[id].setPID(ServoGain.Velocity.value.kp, value)
        elif gain =="kvi":
            self.joints[id].setPID(ServoGain.Velocity.value.ki, value)
        elif gain == "kpp":
            self.joints[id].setPID(ServoGain.Position.value.kp, value)
        elif gain == "kpi":
            self.joints[id].setPID(ServoGain.Position.value.ki, value)
        else:
            print("input argument error!!")

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
            # self.joints[i].setInitial(pos_init=0.5)
    
    def resetServoModel(self):
        # This function is for reset the servo model parameter
        for i,joint in self.joints:
            model_path_name = f"j{i+1}/j{i+1}.sdrv"
            joint.ImportServoModel(saved_model=self.model_path+model_path_name)

    def resetPID(self):
        # This function is for reseting the servo gain
        self.initialize_model()
        

    def compute_thread(self, threadID, i, q_ref, g_tor):
        # pos,vel,acc,tor,pos_err,vel_err = self.joints[threadID](q_ref-self.q_c[0][threadID],g_tor)
        pos,vel,acc,tor,pos_err,vel_err = self.joints[threadID](q_ref,g_tor)
        self.q[i][threadID] = pos 
        self.dq[i][threadID] = vel 
        self.ddq[i][threadID] = acc 
        self.q_pos_err[i][threadID] = pos_err
        self.torque[i][threadID] = tor

    def start(self)->any:
        g_tor = np.zeros(6,dtype=np.float32)
        fric_tor = np.zeros(6,dtype=np.float32)

        self.progress = 0 

        for i, q_ref in enumerate(zip(self.q1_c,self.q2_c,self.q3_c,self.q4_c,self.q5_c,self.q6_c)):
            self.progress = i/self.arr_size*100+1
            
            for idx in range(6):
                pos,vel,acc,tor,pos_err, vel_cmd = self.joints[idx](q_ref[idx],g_tor[idx])
                self.q[i][idx] = pos 
                self.dq[i][idx] = vel 
                self.ddq[i][idx] = acc 
                self.q_pos_err[i][idx] = pos_err
                self.torque[i][idx] = tor

                # thread = threading.Thread(target=self.compute_thread, args=(idx, i, q_ref[idx], g_tor[idx]))
                # self.thread_list.append(thread)
                # self.thread_list[idx].start()

            ## Wait for all thread to finish
            # for thread in self.thread_list:
                # thread.join()
            
            ## Clear the thread list after using it to ensure not to reuse thread again
            # self.thread_list = []

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
            self.tracking_err_pitch.append(self.pitch[i]-self.pitch_c[i])
            self.tracking_err_roll.append(self.roll[i]-self.roll_c[i])
            self.tracking_err_yaw.append(self.yaw[i]-self.yaw_c[i])
            self.tracking_err.append(LA.norm([self.tracking_err_x, self.tracking_err_y, self.tracking_err_z]))

            
            self.contour_err.append(self.computeCountourErr(self.x[i],self.y[i],self.z[i]))
            
            # No need to doing that any more
            # self.x_c[i],self.y_c[i],self.z_c[i] = self.forward_kinematic(
            #                         (self.q1_c[i],self.q2_c[i],self.q3_c[i],
            #                             self.q4_c[i],self.q5_c[i],self.q6_c[i]))
    
    def __call__(self,q_ref:np.ndarray):
        if not isinstance(q_ref,np.ndarray):
            raise TypeError("Input datamust be a Numpy array with size")
        if q_ref.shape != (6,):
            raise ValueError("Input data must have shape (6,)")
        
        g_tor = np.zeros(6,dtype=np.float32)
        fric_tor = np.zeros(6,dtype=np.float32)
        q = np.zeros(6,dtype=np.float32)
        dq = np.zeros(6,dtype=np.float32)
        ddq = np.zeros(6,dtype=np.float32)

                
        for idx in range(6):
            pos,vel,acc,tor,pos_err = self.joints[idx](q_ref[idx],g_tor[idx])
            q[idx] = pos 
            dq[idx] = vel 
            ddq[idx] = acc

        g_tor = self.compute_GTorque(q[1],q[2],q[3],q[4],q[5])
            
        fric_tor = self.compute_friction(q[0],q[1],q[2],
                                        q[3],q[4],q[5]) #TODO

        x,y,z,pitch,roll,yaw = self.forward_kinematic(
                                (q[0],q[1],q[2],q[3],q[4],q[5]))
        
        
        return x,y,z,pitch,roll,yaw

    def save_log(self,save_dir:str=None):
        # self.log_path = save_dir + '/log/'

        # if not os.path.exists(self.log_path):
        #     os.mkdir(self.log_path)

        ### System log ###
        # np.savetxt(self.log_path+'joint_pos_error.txt',self.q_pos_err,delimiter=',',header='Joint1, Joint2, Joint3, Joint4, Joint5, Joint6', fmt='%10f')
        # np.savetxt(self.log_path+'joint_pos.txt',self.q,delimiter=',',header='Joint1, Joint2, Joint3, Joint4, Joint5, Joint6', fmt='%10f')
        # np.savetxt(self.log_path+'tor.txt',self.torque,delimiter=',',header='Joint1, Joint2, Joint3, Joint4, Joint5, Joint6', fmt='%10f')
        # np.savetxt(self.log_path+'contour_error.txt', self.contour_err)
        # np.savetxt(self.log_path+'tracking_error.txt',self.tracking_err, 'fmt=%10f')
        # np.savetxt(self.log_path+'tracking_error_x.txt',self.tracking_err_x, 'fmt=%10f')
        # np.savetxt(self.log_path+'tracking_error_y.txt',self.tracking_err_y, 'fmt=%10f')
        # np.savetxt(self.log_path+'tracking_error_z.txt',self.tracking_err_z, 'fmt=%10f')
        # np.savetxt(self.log_path+'tracking_error_pitch.txt',self.tracking_err_pitch, 'fmt=%10f')
        # np.savetxt(self.log_path+'tracking_error_roll.txt',self.tracking_err_roll, 'fmt=%10f')
        # np.savetxt(self.log_path+'tracking_error_yaw.txt',self.tracking_err_yaw, 'fmt=%10f')

        try:
            connection = pymysql.connect(
                host= "127.0.0.1",  # Localhost IP address
                port= 3305,          # Default MySQL port
                user= "root",        # MySQL root user (caution: use secure credentials)
                password= "Sam512011", # Replace with your actual password
            )

            cursor = connection.cursor()
            cursor.execute("CREATE DATABASE IF NOT EXISTS bw_mismatch_db;")
            cursor.execute("USE bw_mismatch_db;")

            table_name = "bw_mismatch_data"
            sql = f"""CREATE TABLE IF NOT EXISTS {table_name} (
                id INT AUTO_INCREMENT PRIMARY KEY,
                Gain JSON, -- Kp gain of each joints
                Bandwidth JSON,
                contour_err JSON,
                max_bandwidth INT,
                tracking_err_x JSON,
                tracking_err_y JSON,
                tracking_err_z JSON,
                contour_err_img_path VARCHAR(255)
                
            )"""
            cursor.execute(sql)

            gain_json = json.dumps([self.joints[i].pos_amp.kp for i in range(6)])
            bandwidth_json = json.dumps(self.bandwidth)
            max_bandwidth = np.argmax(self.bandwidth)+1
            contour_error_json = json.dumps(self.contour_err)
            tracking_err_x_json = json.dumps(self.tracking_err_x)
            tracking_err_y_json = json.dumps(self.tracking_err_y)
            tracking_err_z_json = json.dumps(self.tracking_err_z)

            sql = "SELECT MAX(id)+1 AS highest_id FROM bw_mismatch_data;"
            cursor.execute(sql)
            current_id = cursor.fetchone()[0]
            # print(current_id)

            fig_path = f"C:\\Users\\Samuel\\Desktop\\mismatch_dataset\\{current_id}.png"
            c_err_img = self.plot_polar(show=False)
            c_err_img.savefig(fig_path)

            sql = """INSERT INTO bw_mismatch_data 
                    (Gain, Bandwidth, max_bandwidth, contour_err, 
                    tracking_err_x, tracking_err_y, tracking_err_z, contour_err_img_path) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"""
            cursor.execute(sql, (gain_json, bandwidth_json, max_bandwidth, contour_error_json, 
                                 tracking_err_x_json, tracking_err_y_json, tracking_err_z_json
                                 ,fig_path))
            connection.commit()
            
        except Exception as ex:
            print(ex)
        # np.savetxt(self.log_path+'joint_vel.txt', self.dq,delimiter=',',header='Joint1, Joint2, Joint3, Joint4, Joint5, Joint6', fmt='%10f')
        # np.savetxt(self.log_path+'joint_acc.txt', self.ddq,delimiter=',',header='Joint1, Joint2, Joint3, Joint4, Joint5, Joint6', fmt='%10f')    

    def pso_tune_gain_update(self):
        """
            this function is seted up temperary for tuning PID gain using PSO algorithm
        """
        g_tor = np.zeros(6,dtype=np.float32)
        fric_tor = np.zeros(6,dtype=np.float32)

        contour_err_sum = 0
        err = 0

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
            
            # self.tracking_err_x.append(self.x[i]-self.x_c[i])
            # self.tracking_err_y.append(self.y[i]-self.y_c[i])
            # self.tracking_err_z.append(self.z[i]-self.z_c[i])
            # self.tracking_err_pitch.append(self.pitch[i]-self.pitch_c[i])
            # self.tracking_err_roll.append(self.roll[i]-self.roll_c[i])
            # self.tracking_err_yaw.append(self.yaw[i]-self.yaw_c[i])
            # self.tracking_err.append(LA.norm([self.tracking_err_x, self.tracking_err_y, self.tracking_err_z]))
            # self.contour_err.append(self.computeCountourErr(self.x[i],self.y[i],self.z[i]))
            c_err = self.computeCountourErr(self.x[i],self.y[i],self.z[i])
            contour_err_sum  = contour_err_sum + c_err**2
            # print( self.computeCountourErr(self.x[i],self.y[i],self.z[i]))
            err = err + ((self.x_c[i]-self.x[i]) + (self.y_c[i]-self.y[i]) + (self.z_c[i]-self.z[i]))**2
        # print(contour_err_sum)
        
        return contour_err_sum


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
        fig,ax = plt.subplots(3,2)

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
        
        for i in range(6):
            ax[i//2,i%2].set_title(f"joint{i+1}")
            ax[i//2,i%2].plot(t,self.q_pos_err[:,i])
            ax[i//2,i%2].grid(True)
            ax[i//2,i%2].set_ylim(mod_q_err[i] - 1.1 * max_range, mod_q_err[i]  + 1.1 * max_range)
            ax[i//2,i%2].set_xlabel("time(s)")
            ax[i//2,i%2].set_ylabel(r"$\theta$(deg)")    


        plt.suptitle('Joint angle error')
        plt.tight_layout()

        if show:
            plt.show()
        return fig

    # create x,y,z plot
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

    def plot_polar(self, show=True):
        #　This function plot the circular trajectory tracking error in polar form

        if self.path_mode==2:
            return
        
        # determine the center of circle
        x_offset = (max(self.x_c) + min(self.x_c))/2 * 1000
        y_offset = (max(self.y_c) + min(self.y_c))/2 * 1000
        z_offset = (max(self.z_c) + min(self.z_c))/2 * 1000

        t = np.array(range(0,self.arr_size))*self.ts

        r = np.zeros(self.arr_size)
        phi = t

        r_c = np.zeros(self.arr_size)
        phi_c = np.zeros(self.arr_size)

        rho = np.zeros(self.arr_size)

        if self.path_mode == 0: # XY circular test

            for i in range(self.arr_size):
                r[i] = math.sqrt((self.x[i]* 1000 - x_offset)**2 + (self.y[i]* 1000 - y_offset)**2)
                phi[i] = t[i]/t[-1] * 2* math.pi

                r_c[i] = math.sqrt((self.x_c[i]* 1000 - x_offset)**2 + (self.y_c[i]* 1000 - y_offset)**2)
                phi_c[i] = t[i]/t[-1] * 2* math.pi

                rho[i] = r[i] - r_c[i]


        elif self.path_mode == 1: # YZ circular test
            for i in range(self.arr_size):
                r[i] = math.sqrt((self.y[i]* 1000  - y_offset)**2 + (self.z[i]* 1000  - z_offset)**2)
                phi[i] = t[i]/t[-1] * 2* math.pi

                r_c[i] = math.sqrt((self.y_c[i]* 1000  - y_offset)**2 + (self.z_c[i]* 1000  - z_offset)**2)
                phi_c[i] = t[i]/t[-1] * 2* math.pi

                rho[i] = r[i] - r_c[i]

        # print(max(rho), min(rho))
        radius_range = max(rho) - min(rho)

        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'},figsize=(3.6,2.7))
        ax.plot(phi, rho)
        ax.set_rmax(max(rho) + 0.2*radius_range)
        ax.set_rmin(min(rho) - 0.2*radius_range)
        # ax.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
        ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
        ax.grid(True)
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig

    def computeCountourErr(self,x,y,z):
        #This function use the mathemetical formula to compute the circular trajectory contouring error
        #Which can reduce the compute complexity from O(n^2) to O(n)
        # print(x,y,z)

        c_err = 0
        x_center = (max(self.x_c) + min(self.x_c)) / 2
        y_center = (max(self.y_c) + min(self.y_c)) / 2
        z_center = (max(self.z_c) + min(self.z_c)) / 2
        # print(self.path_mode)
        if self.path_mode == 0: # XY circular test
            # print(x_center, y_center, z_center)
            # print(x,y,z)
            R = math.sqrt((self.x_c[0] - x_center)**2 + (self.y_c[0] - y_center)**2)
            c_err = math.sqrt((math.sqrt((x-x_center)**2 + (y-y_center)**2) - R)**2 + (z - z_center)**2)   
        elif self.path_mode == 1: # YZ circular test
            R = math.sqrt((self.y_c[0] - y_center)**2 + (self.z_c[0] - z_center)**2)
            c_err = math.sqrt((math.sqrt((y-y_center)**2 + (z-z_center)**2) - R)**2 + (x - x_center)**2)
        elif self.path_mode == 2: # line circular test
            if self.x_c[0] != self.x_c[100]: # X direction line
                c_err = math.sqrt((y - y_center)**2 + (z - z_center)**2)
            elif self.y_c[0] != self.y_c[100]: # Y direction line
                c_err = math.sqrt((x - x_center)**2 + (z - z_center)**2)
            elif self.z_c[0] != self.z_c[100]: # Z direction line
                c_err = math.sqrt((x - x_center)**2 + (y - y_center)**2)
        # print(f'{x,y,z}: {c_err}')

        return c_err
    
    def computeTrackErr(self,x,y,z,psi, phi,theta):
        t_err_x = x - self.x_c
        t_err_y = y - self.y_c
        t_err_z = z - self.z_c
        t_err_psi = psi - self.psi_c
        t_err_phi = phi - self.phi_c
        t_err_theta = theta - self.theta_c

        t_err = math.sqrt(t_err_x**2 + t_err_y**2 + t_err_z**2)

        return t_err, t_err_x, t_err_y, t_err_z, t_err_psi, t_err_phi, t_err_theta

    
    def freq_response(self, fs=2000,f0=0.1,f1=50,t0=0,t1=1,a0=0.5,a1=0.001, show=True):
        '''
        determine the frequency response of system:
        fs: sampling rate
        f0: start freq
        f1: end freq
        t0: start time
        t1: duration of chirp
        a0: start amplitude
        a1: end amplitude
        '''
        t = np.linspace(t0,t1,fs*(t1-t0),endpoint=False)

        # generate chirp sine signal
        amp_decay = a0 + (a1 - a0)/(t1-t0)*t
        f = f0 + (f1-f0)/(t1-t0)*t
        chirp_sine = amp_decay*np.sin(2*np.pi*f*t)


        shape = (6,6,len(t))

        q_c = np.zeros(shape)

        # initialize q_c with initial joint position
        for i in range(6):
            for j in range(6):
                q_c[i][j] = [self.q_c[0][j] for _ in range(len(t))]
                if i==j:
                    q_c[i][j] = q_c[i][j] + chirp_sine
                    
                    
        q = np.zeros(6)

        self.fig, self.ax = plt.subplots(2, 1, figsize=(4.3, 4.1))

        self.bandwidth = []

        #determine frequency response of each joint
        for joint_num in range(6):
            output = []
            g_tor = np.zeros(6,dtype=np.float32)
            fric_tor =  np.zeros(6,dtype=np.float32)
            for time in range(len(t)):
                for idx in range(6):
                
                    q[idx],dq,ddq,self.__tor_internal, pos_err, vel_cmd = self.joints[idx](q_c[joint_num][idx][time],g_tor[idx])
                    if joint_num==idx:
                        output.append(q[idx])    

                g_tor = self.compute_GTorque(q[1],q[2],q[3],q[4],q[5])
                fric_tor = self.compute_friction(q[0],q[1],q[2],q[3],q[4],q[5])

            # fig, ax = plt.subplots()
            # ax.plot(q_c[joint_num][joint_num], label='q_c')
            # ax.plot(output,label='q_s')
            # ax.set_xlabel('Index')
            # ax.set_ylabel('Values')
            # ax.legend()
            # plt.show()

            yf = np.fft.fft(output)
            xf = np.fft.fft(q_c[joint_num][joint_num])

            freqs = np.fft.fftfreq(len(yf/xf)) * fs
            mag = np.abs(yf/xf)

            # Finding the index of the frequency where magnitude reaches -3 dB
            dB_threshold = -3
            
            
            diff = 20*np.log10(mag[:len(mag)//2]) - dB_threshold

            for index, f in enumerate(diff):
                if  f < 0:
                    break
           
            self.bandwidth.append(freqs[index])

            phase = np.angle(yf/xf)
            
        if show:
            self.ax[0].semilogx(freqs[:len(freqs)//2], 20*np.log10(mag[:len(mag)//2]), label=f"joint {joint_num+1} - {freqs[index]:.2f} Hz")   
            
            self.ax[0].set_xlabel('Frequency [Hz]')
            self.ax[0].set_ylabel('Magnitude [dB]')
            self.ax[0].grid(True)
            self.ax[0].set_xlim([f0, f1])
            self.ax[1].semilogx(freqs[:len(freqs)//2], phase[:len(phase)//2])
            self.ax[1].set_xlabel('Frequency [Hz]')
            self.ax[1].set_ylabel('Phase [rad]')
            
            self.ax[1].grid(True)
            self.ax[1].set_xlim([f0, f1])
            self.ax[0].legend()

            plt.show()
        else:
            plt.close(self.fig)

        return self.fig

    def plot_torq(self):

        plt.plot( )
        for i in range(6):
            plt.plot(self.time,self.torque[:,i], label=f'link {i+1}')
        plt.title('torque')
        plt.grid(True)
        plt.legend()
        plt.show()
        