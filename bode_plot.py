############ bode plot ###########
from libs.RobotManipulators import RT605_710

from libs.ServoMotor import ServoMotor
from libs.type_define import *
import libs.ControlSystem as cs
import libs.ServoDriver
from libs.ServoDriver import JointServoDrive
from libs.ForwardKinematic import FowardKinematic
from libs.rt605_Gtorq_model import RT605_GTorq_Model
from libs.rt605_Friction_model import RT605_Friction_Model

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from rt605 import RT605

# gain abs(Y/X)
# phase angle(Y/X)
# frequency range (0.001~ 50Hz)
# input chirp sine

#define frequency range (from(f0~f1)=0.01~100Hz) sampling rate(fs)=1000

class Freq_Response():
    def __init__(self) -> None:
        self.fs = None
        self.t = []
        self.f = []
        self.chirp_sine = []
        self.f0 = None   # start frequency
        self.f1 = None   # end frequecy
        self.t1 = None   # duration of chirp
        self.t0 = None   # start time
        self.amp = None  # chirp sine amplitude 
        self.bias = None   # chirp sine bias
        # log:
        self.log_command = []
        self.log_actual = []
        self.bandwidth = 0
        

        # self.compute_GTorque = RT605_GTorq_Model()
        # self.compute_friction = RT605_Friction_Model()
        # self.q_init =  (90,52.839000702,0.114,0,-52.952999115,0) # initial angle of each joint

    def setupChirpSine(self, fs, f:tuple, t:tuple, amp:float, bias:float, show=False) ->None:
        self.fs = fs
        self.f0, self.f1 = f[0], f[1]
        self.t0, self.t1 = t[0], t[1]
        self.amp = np.deg2rad(amp)
        self.bias = np.deg2rad(bias)
        
        ts = 1/self.fs
        self.t = np.arange(self.t0, self.t1, ts)
        self.f = np.linspace(self.f0, self.f1, len(self.t))
        
        self.chirp_sine = np.rad2deg(self.bias + self.amp*np.sin(2*np.pi*self.f*self.t))
        
        if show == True:
            plt.figure()
            plt.plot(self.t, self.chirp_sine)
            plt.grid()
            plt.show()

    def start(self, robot:RT605, q_init:tuple, index:int):
        pass
    # def __call__(self,  motors:JointServoDrive) -> Any:
    #     self.fig, self.ax = plt.subplots(2, 1, figsize=(8, 6))
    #     for idx, motor in enumerate(motors):
    #         amp_decay = self.a0 + (self.a1-self.a0)/(self.t1-self.t0)*self.t
    #         f = self.f0 + (self.f1-self.f0)/(self.t1-self.t0)*self.t
    #         chirp_sin = amp_decay*np.sin(2*np.pi*f*self.t)


    #         output = np.zeros(chirp_sin.shape[0])
    #         for i, input in enumerate(chirp_sin):
    #             q,dq,ddq,self.__tor_internal, pos_err,vel_err = motor(input,)
    #             output[i] = q
                
                

    #             # output[i] = pos
            
                


    #         yf = np.fft.fft(output)
    #         xf = np.fft.fft(chirp_sin)

    #         freqs = np.fft.fftfreq(len(yf/xf)) * self.fs
    #         mag = np.abs(yf/xf)

    #         # Finding the index of the frequency where magnitude reaches -3 dB
    #         dB_threshold = -3
            
            
    #         diff = 20*np.log10(mag[:len(mag)//2]) - dB_threshold

    #         for index, f in enumerate(diff):
    #             if  f < 0:
    #                 break
    #         self.bandwidth = freqs[index]

    #         phase = np.angle(yf/xf)
            

    #         self.ax[0].semilogx(freqs[:len(freqs)//2], 20*np.log10(mag[:len(mag)//2]), label=f"joint {idx+1} - {self.bandwidth} Hz")   
            
    #         self.ax[0].set_xlabel('Frequency [Hz]')
    #         self.ax[0].set_ylabel('Magnitude [dB]')
    #         self.ax[0].grid(True)
    #         self.ax[0].set_xlim([self.f0, self.f1])
    #         self.ax[1].semilogx(freqs[:len(freqs)//2], phase[:len(phase)//2])
    #         self.ax[1].set_xlabel('Frequency [Hz]')
    #         self.ax[1].set_ylabel('Phase [rad]')
            
    #         self.ax[1].grid(True)
    #         self.ax[1].set_xlim([self.f0, self.f1])
    #         self.ax[0].legend()
    #     plt.show()
    #     return self.fig






if __name__ == "__main__":
    
    fa = Freq_Response()
    fa.setupChirpSine(fs=2000, f = (0.001, 100), t = (0.0, 3), amp=0.02, bias=90.0, show=True)