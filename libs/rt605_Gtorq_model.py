from typing import Any
from libs.RobotManipulators import RT605_710
import libs.ServoDriver as dev
import numpy as np
from threading import Thread
from libs.ServoMotor import ServoMotor
from libs.type_define import *
from libs import ControlSystem as cs
from libs import ServoDriver
from libs.ForwardKinematic import FowardKinematic
import json

class RT605_GTorq_Model():
    def __init__(self, en=False) -> None:
        self.enabled = en
        self.m2=8.663838679819902
        self.m3=3.666344308215677
        self.m4=3.615885748802093
        self.m5= 1.030957155800785
        self.m6= 1.516126236073167
        self.rcx2 = -0.144212674049077
        self.rcy2 = 0.030223509965502


        self.rcx3 = -0.033723402680331
        self.rcz3 = 0.052721699263344

        self.rcx4 = -0.001489966345731
        self.rcy4 = 0.156987050585444
        self.rcz4 =-0.001747720466200

        self.rcx5 = -0.005033704747790
        self.rcy5 = -0.001191062175159
        self.rcz5 = 0.074853989870876

        self.rcx6 = 8.855201349532231e-04
        self.rcy6 = -2.306859519121158e-04
        self.rcz6 =0.045616506355611
    
    def enable_Gtorq(self,en:bool):
        self.enabled = en

    def __call__(self, q2, q3, q4, q5, q6) -> Any:
        if self.enabled==True:
            t2 = np.cos(q2)
            t3 = np.cos(q4)
            t4 = np.cos(q5)
            t5 = np.cos(q6)
            t6 = np.sin(q4)
            t7 = np.sin(q5)
            t8 = np.sin(q6)
            t9 = q2+q3
            t10 = q4+q5
            t11 = q4+q6
            t16 = -q5
            t17 = -q6
            t12 = np.cos(t9)
            t13 = np.cos(t10)
            t14 = np.sin(t9)
            t15 = np.sin(t10)
            t18 = q4+t16
            t19 = q4+t17
            t20 = np.cos(t18)
            t21 = np.sin(t18)
            t22 = self.m3*t12*3.924e-1
            t23 = self.m4*t12*3.924e-1
            t24 = self.m5*t12*3.924e-1
            t25 = self.m6*t12*3.924e-1
            t26 = self.m3*self.rcx3*t12*(9.81e+2/1.0e+2)
            t27 = self.m4*self.rcy4*t14*(9.81e+2/1.0e+2)
            t28 = self.m3*self.rcz3*t14*(9.81e+2/1.0e+2)
            t30 = self.m4*self.rcx4*t3*t12*(9.81e+2/1.0e+2)
            t31 = self.m5*self.rcy5*t6*t12*(9.81e+2/1.0e+2)
            t32 = self.m4*self.rcz4*t6*t12*(9.81e+2/1.0e+2)
            t33 = self.m5*self.rcz5*t4*t14*(9.81e+2/1.0e+2)
            t34 = self.m6*self.rcz6*t4*t14*(9.81e+2/1.0e+2)
            t35 = self.m5*self.rcx5*t7*t14*(9.81e+2/1.0e+2)
            t36 = self.m4*t14*3.31578
            t37 = self.m5*t14*3.31578
            t38 = self.m6*t14*3.31578
            t42 = self.m6*self.rcx6*t5*t7*t14*(9.81e+2/1.0e+2)
            t43 = self.m6*self.rcy6*t7*t8*t14*(9.81e+2/1.0e+2)
            t45 = self.m6*t4*t14*8.48565e-1
            t29 = -t27
            t39 = -t31
            t40 = -t32
            t41 = -t35
            t44 = -t42
            G = np.array([0,
                        t22+t23+t24+t25+t26+t28+t29+t30+t33+t34+t36+t37+t38+t39+t40+t41+t43+t44+t45+self.m2*t2*3.3354+self.m3*t2*3.3354+self.m4*t2*3.3354+self.m5*t2*3.3354+self.m6*t2*3.3354-self.m2*self.rcy2*np.sin(q2)*(9.81e+2/1.0e+2)+self.m2*self.rcx2*t2*(9.81e+2/1.0e+2)+self.m6*t12*t15*4.242825e-1-self.m6*t12*t21*4.242825e-1+self.m5*self.rcx5*t12*t13*(9.81e+2/2.0e+2)+self.m5*self.rcx5*t12*t20*(9.81e+2/2.0e+2)+self.m5*self.rcz5*t12*t15*(9.81e+2/2.0e+2)+self.m6*self.rcz6*t12*t15*(9.81e+2/2.0e+2)-self.m5*self.rcz5*t12*t21*(9.81e+2/2.0e+2)-self.m6*self.rcz6*t12*t21*(9.81e+2/2.0e+2)+self.m6*self.rcx6*t12*np.cos(t11)*(9.81e+2/2.0e+2)-self.m6*self.rcx6*t12*np.cos(t19)*(9.81e+2/2.0e+2)-self.m6*self.rcy6*t12*np.sin(t11)*(9.81e+2/2.0e+2)-self.m6*self.rcy6*t12*np.sin(t19)*(9.81e+2/2.0e+2)+self.m6*self.rcx6*t5*t12*t13*(9.81e+2/2.0e+2)+self.m6*self.rcx6*t5*t12*t20*(9.81e+2/2.0e+2)-self.m6*self.rcy6*t8*t12*t13*(9.81e+2/2.0e+2)-self.m6*self.rcy6*t8*t12*t20*(9.81e+2/2.0e+2),
                        t22+t23+t24+t25+t26+t28+t29+t30+t33+t34+t36+t37+t38+t39+t40+t41+t43+t44+t45+self.m6*t3*t7*t12*8.48565e-1+self.m5*self.rcx5*t3*t4*t12*(9.81e+2/1.0e+2)-self.m6*self.rcx6*t6*t8*t12*(9.81e+2/1.0e+2)-self.m6*self.rcy6*t5*t6*t12*(9.81e+2/1.0e+2)+self.m5*self.rcz5*t3*t7*t12*(9.81e+2/1.0e+2)+self.m6*self.rcz6*t3*t7*t12*(9.81e+2/1.0e+2)+self.m6*self.rcx6*t3*t4*t5*t12*(9.81e+2/1.0e+2)-self.m6*self.rcy6*t3*t4*t8*t12*(9.81e+2/1.0e+2),
                        t14*(self.m4*self.rcx4*t6*2.0e+3+self.m5*self.rcy5*t3*2.0e+3+self.m4*self.rcz4*t3*2.0e+3+self.m6*t6*t7*1.73e+2+self.m5*self.rcx5*t4*t6*2.0e+3+self.m6*self.rcx6*t3*t8*2.0e+3+self.m6*self.rcy6*t3*t5*2.0e+3+self.m5*self.rcz5*t6*t7*2.0e+3+self.m6*self.rcz6*t6*t7*2.0e+3+self.m6*self.rcx6*t4*t5*t6*2.0e+3-self.m6*self.rcy6*t4*t6*t8*2.0e+3)*(-4.905e-3),
                        self.m6*(t7*t12*8.65e-2+self.rcz6*t7*t12+t3*t4*t14*8.65e-2+self.rcx6*t4*t5*t12-self.rcy6*t4*t8*t12+self.rcz6*t3*t4*t14-self.rcx6*t3*t5*t7*t14+self.rcy6*t3*t7*t8*t14)*(9.81e+2/1.0e+2)+self.m5*(self.rcx5*t4*t12+self.rcz5*t7*t12-self.rcx5*t3*t7*t14+self.rcz5*t3*t4*t14)*(9.81e+2/1.0e+2),
                        self.m6*(self.rcx6*(t5*t6*t14+t7*t8*t12+t3*t4*t8*t14)+self.rcy6*(t5*t7*t12-t6*t8*t14+t3*t4*t5*t14))*(-9.81e+2/1.0e+2)]).reshape(-1,1)
            # tor = G.transpose()
            # tor1 = tor(1)
            # tor2 = tor(2)
            # tor3 = tor(3)
            # tor4 = tor(4)
            # tor5 = tor(5)
            # tor6 = tor(6)
        else:
            G = np.zeros(6)
            # tor1 = 0
            # tor2 = 0
            # tor3 = 0
            # tor4 = 0
            # tor5 = 0
            # tor6 = 0
        return G

       

        
        