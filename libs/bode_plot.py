############ bode plot ###########
from libs.RobotManipulators import RT605_710
import libs.ServoDriver as dev
import numpy as np
from threading import Thread
from libs.ServoMotor import ServoMotor
from libs.type_define import *
from libs import ControlSystem as cs
from libs import ServoDriver
from libs.ServoDriver import JointServoDrive
from libs.ForwardKinematic import FowardKinematic
from libs.rt605_Gtorq_model import RT605_GTorq_Model
from libs.rt605_Friction_model import RT605_Friction_Model

from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# gain abs(Y/X)
# phase angle(Y/X)
# frequency range (0.001~ 50Hz)
# input chirp sine

#define frequency range (from(f0~f1)=0.01~100Hz) sampling rate(fs)=1000

class Freq_Response():
    def __init__(self,fs=1000,f0=0.1,f1=100,t1=1,t0=0,a0=1,a1=0.01) -> None:
        self.fs = fs
        self.t = np.linspace(0,1,fs, endpoint=False)
        self.f0 = f0   # start frequency
        self.f1 = f1   # end frequecy
        self.t1 = t1   # duration of chirp
        self.t0 = t0   # start time
        self.a0 = a0   # start amplitude 
        self.a1 = a1   # end amplitude
        self.bandwidth = 0

        self.compute_GTorque = RT605_GTorq_Model()
        self.compute_friction = RT605_Friction_Model()
        self.q_init =  (90,52.839000702,0.114,0,-52.952999115,0) # initial angle of each joint

        


    def __call__(self,  motors:JointServoDrive) -> Any:
        self.fig, self.ax = plt.subplots(2, 1, figsize=(8, 6))
        for idx, motor in enumerate(motors):
            amp_decay = self.a0 + (self.a1-self.a0)/(self.t1-self.t0)*self.t
            f = self.f0 + (self.f1-self.f0)/(self.t1-self.t0)*self.t
            chirp_sin = amp_decay*np.sin(2*np.pi*f*self.t)


            output = np.zeros(chirp_sin.shape[0])
            for i, input in enumerate(chirp_sin):
                q,dq,ddq,self.__tor_internal, pos_err,vel_err = motor(input,)
                output[i] = q
                
                

                # output[i] = pos
            
                


            yf = np.fft.fft(output)
            xf = np.fft.fft(chirp_sin)
            
            mag = np.abs(yf/xf)
            phase = np.angle(yf/xf, deg=True)

            # Convert frequency to Hz
            freqs = np.fft.fftfreq(len(yf/xf)) * self.fs
            
            # Finding the index of the frequency where magnitude reaches -3 dB
            dB_threshold = -3
            
            
            diff = 20*np.log10(mag[:len(mag)//2]) - dB_threshold

            for index, f in enumerate(diff):
                if  f < 0:
                    break
            self.bandwidth = freqs[index]

            
            

            self.ax[0].semilogx(freqs[:len(freqs)//2], 20*np.log10(mag[:len(mag)//2]), label=f"joint {idx+1} - {self.bandwidth} Hz")
            self.ax[0].set_xlabel('Frequency [Hz]')
            self.ax[0].set_ylabel('Magnitude [dB]')
            self.ax[0].grid(True)
            self.ax[0].set_xlim([self.f0, self.f1])
            self.ax[0].legend()

            self.ax[1].semilogx(freqs[:len(freqs)//2], phase[:len(phase)//2])
            self.ax[1].set_xlabel('Frequency [Hz]')
            self.ax[1].set_ylabel('Phase [rad]')
            self.ax[1].grid(True)
            self.ax[1].set_xlim([self.f0, self.f1])
            
        plt.show()
        return self.fig







