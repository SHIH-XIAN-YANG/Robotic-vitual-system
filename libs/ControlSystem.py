from libs.type_define import*
import numpy as np
from scipy import signal

#
class ForwardLoop1D:
    # -------------------------------------------------------------
    # Static Attributes: 
    # -------------------------------------------------------------
    ts:np.float32 = 0.001 # sampling time (unit: second) To be change!!!
    loop_gain:np.float32 = 1.0
    __id = int() # 編號該模組之 ID，用為外部第三方模組抓取此模組輸出 y 的值
    # For S-domain simulation:
    poles = []
    zeros = []
    # For Time-domain simulation:
    y:np.float32 = None # 當前模組的輸出值
    
    # -------------------------------------------------------------
    def __init__(self, id:int):
        self.setID(id)
        self.y = 0.0
    #
    def setGain(self,gain:np.float32):
        self.loop_gain = gain
    #
    def gain(self):
        return self.loop_gain
    #
    def setID(self, _set_id:int):
        self.__id = _set_id
    #
    def ID(self):
        return self.__id
    #
    def value(self):
        return self.y
    
    def setSamplingTime(self, ts:np.float32):
        self.ts = ts
    #
    def __call__(self, x:np.float32):
        return self.loop_gain*x
#
class Ratio(ForwardLoop1D):
    def __init__(self, id:int):
        super().__init__(id)
        self.setID(id)
        self.setGain(gain=1.0)
    #
    def setRatio(self,num:np.float32, den:np.float32):
        self.setGain(gain=num/den)
    #
    def __call__(self, x:np.float32, inverse:bool=False):
        if inverse == True:
            self.y = x/self.gain()
        else:
            self.y = x*self.gain()
        return self.y
    
class Reducer(Ratio): #減速比
    def __init__(self, id:int=0):
        super().__init__(id=id)
    
    
#
FIR_LOWPASS:int = 0
IIR_LOWPASS:int = 1

class Exponential_delay(ForwardLoop1D):
    time_const:np.float32 = None
    unit:str = None
    __k:np.float32 = None # filter gain


    def __init__(self, id: int,k=1, tau=1):
        self.setID(id)
        self.__k = k


    def setup(self, tc:np.float32):
        self.time_const = tc

    def __call__(self, x: np.float32,t:np.float32):
        y = self.__k*(1-np.exp(-1*t/self.time_const))
        return y
        

class LowPassFilter1D(ForwardLoop1D):
    __type:int = None
    __order:np.uint32 = None
    __fc:np.float32 = None
    unit:str = None
    __coefficients = None

    #
    def __init__(self, id:int):
        self.setID(id)
        # self.ts = np.float32() # sampling time (unit: second)
        # self.zeros = []
        # self.poles = []
        # self.x_buf = []
        # self.y_buf2 = []
    #
    def calculate_coefficients(self):
        # Calculate filter coefficients using window method (Hamming window)
        coefficients = np.sinc(2 * self.__fc * (np.arange(self.__order) - (self.__order - 1) / 2))
        coefficients *= np.hamming(self.__order)
        coefficients /= np.sum(coefficients)
        return coefficients
    
    def Setup(self, type:int, order:int, fc:np.float32):
        '''
        [I/P]:
            - type: ControlSystem.FIR_LOWPASS  或  ControlSystem.IIR_LOWPASS
        '''
        self.__type = type
        self.__order = order
        self.__fc = fc

        self.x_buf = np.zeros(self.__order)
        self.__coefficients = self.calculate_coefficients()
    
    
    def get_fc(self):
        return self.__fc
    
    def set_fc(self, fc):
        self.__fc = fc

    def get_order(self):
        return self.__order
    
    def set_order(self, order):
        self.__order = order
    #
    def __call__(self, x:np.float32):
        y:np.float32 = 0.0

        self.x_buf = np.roll(self.x_buf, 1)
        self.x_buf[0] = x
        y = np.sum(self.x_buf * self.__coefficients)
        return y
#
PARRALLEL_PID:bool = False
SERIAL_PID:bool    = True

class PID_Controller1D(ForwardLoop1D):
    # -------------------------------------------------------------
    # Static Attributes: 
    # -------------------------------------------------------------
    pid_en:list   = [0, 0, 0]
    ps:bool = None
    kp:np.float32 = None
    ki:np.float32 = None
    kd:np.float32 = None

    kp_unit:str = None
    ki_unit:str = None
    kd_unit:str = None

    # -------------------------------------------------------------
    def __init__(self, id:int, pid_type:PidType=PidType.PID, ps_sel:bool=PARRALLEL_PID):
        '''
        [I/P]:
        - pid_type: (default: PidType.PID)
            選擇 PID 的形式，分別為：
            (a) PidType.P (b) PidType.PI (c) PidType.PID。
        - ps_sel: (default: PARRALLEL_PID)
            選擇 ControlSystem.PARRALLEL_PID 或是  ControlSystem.SERIAL_PID。
        '''
        #super().__init__()

        self.setID(id)
        for i in range(3):
            t1 = 0b0001 << i
            if pid_type.value & t1 == t1:
                self.pid_en[i] = 1
        self.ps = ps_sel
        self.e_k1:np.float32 = 0.0
        self.e_k2:np.float32 = 0.0
        self.y_k1:np.float32 = 0.0 # Initial value
        
    #
    def Setup(self):
        pass
    #
    def __call__(self, e:np.float32):
        

        if self.ps == PARRALLEL_PID:
            #print("PARRALLEL_PID")
            self.y = self.y_k1 + self.pid_en[0]*self.kp*(e - self.e_k1) + \
                                 self.pid_en[1]*self.ki*(e*self.ts) + \
                                 self.pid_en[2]*self.kd*(e-2*self.e_k1+self.e_k2)/self.ts
        elif self.ps == SERIAL_PID:
            #print("SERIAL_PID")
            if self.ki==0.0:
                self.y = self.y_k1 + self.pid_en[0]*self.kp*((e - self.e_k1) \
                               + self.pid_en[2]*self.kd*(e - 2*self.e_k1 + self.e_k2)/self.ts)
            else:
                self.y = self.y_k1 + self.pid_en[0]*self.kp*((e - self.e_k1) \
                               + self.pid_en[1]/self.ki*e*self.ts \
                               + self.pid_en[2]*self.kd*(e - 2*self.e_k1 + self.e_k2)/self.ts)
        # Update previous error values
        self.e_k2 = self.e_k1
        self.e_k1 = e
                
        # Update previous output value
        self.y_k1 = self.y
        
        #print(type(self.y),type(self._ForwardLoop1D__loop_gain))
        return self.y*self.loop_gain
#
class Node:
    __id = int()
    def __init__(self, id:int):
        self.setID(id)
        # self.input_num:int = 0
        # self.input_id_list = []
        # self.input_polarity = []
        # self.y:np.float32 = 0.0 
    #
    # def AddInput(self, loop_id:int, polarity:int=1):
    #     '''
    #         [I/P]:
    #         - loop_id: 要連結來自於編號 loop_id 的 ForwardLoop 之輸出 y。
    #         - polarity: (default: +1)
    #             輸入來源的極性，分別 +1 或 -1 。
    #     '''
    #     self.input_num += 1
    #     self.input_id_list.append(loop_id)
    #     self.input_polarity.append(polarity)
    # #
    def setID(self, _id:int):
        self.__id = _id
    #
    def ID(self):
        return self._id
    #
    def __call__(self, *args):
        return sum(args)
#  
class Differentiation(ForwardLoop1D):
    __id:int = None
    __first_step:bool = None
    ts:np.float32 = None
    x_k_1:np.float32 = None
    y:np.float32 = None
    #
    def __init__(self, id:int, ts:np.float32=0.001, x_k_1:np.float32=0.0):
        self.setID(id)
        self.Setup(ts=ts, x_k_1=x_k_1)
        self.__first_step = False
    #
    def Setup(self, ts:np.float32, x_k_1:np.float32):
        self.ts = ts
        self.x_k_1 = x_k_1
    #
    def reset(self):
        self.x_k_1 = 0.0
        self.__first_step = False
    #
    def setID(self, _id:int):
        self.__id = _id
    #
    def ID(self):
        return self._id
    #
    def __call__(self, x:np.float32):
        if self.__first_step == False:
            self.y = 0.0
            self.__first_step = True
        else:
            self.y = (x - self.x_k_1)/self.ts
        self.x_k_1 = x 
        return self.y
#
INTEGRAL_METHOD_TRAPZOIDAL:bool = False
INTEGRAL_METHOD_RIEMANN:bool = True

class Integration(ForwardLoop1D):
    __id:int = None
    ts:np.float32 = None
    x_k_1:np.float32 = None
    y:np.float32 = None
    integral_method:bool = None
    #
    def __init__(self, id:int, ts:np.float32=0.001, integral_method:bool=INTEGRAL_METHOD_TRAPZOIDAL, x_k_1:np.float32=0.0, y0:np.float32=0.0):
        self.setID(id)
        self.Setup(ts=ts, integral_method=integral_method, x_k_1=x_k_1, y0=y0)
    #
    def Setup(self, ts:np.float32, integral_method:bool, x_k_1:np.float32, y0:np.float32):
        self.integral_method = integral_method
        self.x_k_1 = x_k_1
        self.y = y0
        self.ts = ts
    #
    def Reset(self):
        self.x_k_1 = 0.0
        self.y = 0.0
    #
    def setID(self, _id:int):
        self.__id = _id
    #
    def ID(self):
        return self._id
    #
    def __call__(self, x:np.float32):
        if self.integral_method == INTEGRAL_METHOD_RIEMANN:
            self.y += x*self.ts
        elif self.integral_method == INTEGRAL_METHOD_TRAPZOIDAL:
            self.y += 0.5*(self.x_k_1 + x)*self.ts
        self.x_k_1 = x
        return self.y
#
class Limitation(ForwardLoop1D):
    __id:int = None
    __enabled:bool = None
    y:np.float32 = None
    H_lim:np.float32 = None
    L_lim:np.float32 = None
    unit:str = None
    
    def __init__(self, id:int):
        self.setID(id)
    #
    def setID(self, _id:int):
        self.__id = _id
    #
    def ID(self):
        return self._id
    #
    def Enable(self, en:bool=True):
        self.__enabled = en
    #
    def setLimitation(self, lim:tuple):
        self.H_lim = max(lim)
        self.L_lim = min(lim)
    #
    def __call__(self, x:np.float32):
        if self.__enabled == False:
            self.y = x
        else:
            if x >= self.H_lim:
                self.y = self.H_lim
                print('value over upper bound!!')
            elif x <= self.L_lim:
                print('value under lower bound!!')
                self.y = self.L_lim
            else:
                self.y = x
        return self.y
    

#
'''
class SequentialModel1D:
    # -------------------------------------------------------------
    # Static Attributes: 
    # -------------------------------------------------------------
    ts:np.float32 = 0.0 # sampling time (unit: second)
    y_list = []
    # -------------------------------------------------------------
    def __init__(self):
        pass
    #
    def LinkLoopPath(self):
        pass
    #
    def BuilModel(self):
        pass
    #
    def Reset(self):
        pass
'''