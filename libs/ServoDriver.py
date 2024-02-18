import numpy as np
from threading import Thread
from libs.ServoMotor import ServoMotor
from libs.type_define import*
from libs import ControlSystem as cs
import json
import time
import re
import os 

class JointServoDrive:
    # -------------------------------------------------------------
    # Static Attributes: 
    # -------------------------------------------------------------
    module:list = []
    __joint_ID:int = None
    __id:int = None
    ts:np.float32 = None
    __pos:np.float32 = None # EX: position in physical unit
    __vel:np.float32 = None
    __acc:np.float32 = None
    __tor:np.float32 = None
    __pos_err:np.float32 = None
    
    __nonlinear_enabled:bool = None
    # servo drive internal status:
    __pos_internal:np.float32 = None # EX: pulse unit
    __vel_internal:np.float32 = None
    __acc_internal:np.float32 = None
    __tor_internal:np.float32 = None
    en_cmd_filters:bool = False
    pos_err_internal:np.float32 = None
    vel_err_internal:np.float32 = None
    vel_cmd_internal:np.float32 = None
    tor_cmd_internal:np.float32 = None
    vel_cmdf_internal:np.float32 = None # filtered command
    tor_cmdf_internal:np.float32 = None # filtered command

    model_path:str = None # save servo model parameter
    # -------------------------------------------------------------
    def __init__(self, id:int, saved_model:str):
        # print('test test test' + os.getcwd())
        # print(saved_model)
        self.BuildModel()
        self.__id = id
        self.__flag = False
        self.__nonlinear_enabled = False
        if saved_model[-5:] != ".sdrv":
            print("Error syntax.")
        else:
            self.ImportServoModel(saved_model=saved_model)
            
            self.__joint_ID = int(re.search(r'\d', saved_model).group())
            
            #self.ImportGainSetting(saved_file=saved_model)
    #
    def setID(self, _set_id:int):
        self.__id = _set_id
    #
    def ID(self):
        return self.__id
    #
    def BuildModel(self):
        # Position Loop:
        
        self.pos_node = cs.Node(id=1)
        self.pos_amp = cs.PID_Controller1D(id=2, pid_type=PidType.P, ps_sel=cs.SERIAL_PID)
        self.vel_cmp = cs.Node(id=3)
        self.vel_cmd_lim = cs.Limitation(id=4)
        self.vel_cmd_filter = cs.ExponentialDecay(id=5)
        # Velocity Loop:
        self.vel_node = cs.Node(id=6)
        self.vel_amp = cs.PID_Controller1D(id=7, pid_type=PidType.PI, ps_sel=cs.SERIAL_PID)
        self.tor_cmp = cs.Node(id=8)
        self.tor_cmd_lim = cs.Limitation(id=9)
        self.tor_cmd_filter = cs.LowPassFilter1D(id=10)
        # Current Loop:
        self.tor_amp = cs.Ratio(id=11)
        self.g_tor_node = cs.Node(id=25)
        # Motor Model:
        self.motor = ServoMotor(id=13)
        self.xr_unit_to_internal = cs.Ratio(id=0)
        self.pos_unit_to_internal = cs.Ratio(id=21)
        self.vel_unit_to_internal = cs.Ratio(id=22)
        self.acc_unit_to_internal = cs.Ratio(id=23)
        self.tor_unit_to_internal = cs.Ratio(id=12)
        self.reducer = cs.Reducer()
    #
    def ImportServoModel(self, saved_model:str):
        # print('ImportServoModel: '+saved_model)
        with open(saved_model) as fs:
            drv_info = json.loads(fs.read())
            self.model_save_path = drv_info["model_save_path"]
            self.PPR = drv_info["PPR"]
            self.phy_unit = drv_info["physical_unit"]
            self.internal_unit = drv_info["internal_unit"]
            self.rated_tor = drv_info["rated_torque"]["value"]
            self.ts = drv_info["sampling_time"]
            self.reducer.setRatio(1.0, drv_info["gear_ratio"])
            

            if self.phy_unit == "rad":
                self.PPU = self.PPR/(2*np.pi)
            elif self.phy_unit == "degree":
                self.PPU = self.PPR/360.0
        self.model_path = saved_model
        self.ImportGainSetting(saved_file=saved_model + self.model_save_path + drv_info["gain_setting"])
        self.motor.ImportMotorModel(saved_model=saved_model + self.model_save_path + drv_info["motor_model"])
        self.xr_unit_to_internal.setGain(self.PPU)
        self.pos_unit_to_internal.setGain(self.PPU) # from internal position value to actual position value
        self.vel_unit_to_internal.setGain(self.PPU)
        self.acc_unit_to_internal.setGain(self.PPU)
        self.tor_unit_to_internal.setGain(1/self.rated_tor*1000)
        
        #   velocity command limit: rpm--> pulse/s
        self.vel_cmd_lim.setLimitation((self.vel_cmd_lim.H_lim/60*self.PPR, self.vel_cmd_lim.L_lim/60*self.PPR))
        
    #
    def ImportGainSetting(self, saved_file:str):
        
        with open(saved_file) as fs:
            k = json.loads(fs.read())
            # print(saved_file)
            # PID gains:
            self.pos_amp.kp = k["position_loop"]["KP"]["value"]
            self.pos_amp.ki = k["position_loop"]["KI"]["value"]
            self.pos_amp.kd = k["position_loop"]["KD"]["value"]
            self.vel_amp.kp = k["velocity_loop"]["KP"]["value"]
            self.vel_amp.ki = k["velocity_loop"]["KI"]["value"]
            self.vel_amp.kd = k["velocity_loop"]["KD"]["value"]

            self.pos_amp.kp_unit = k["position_loop"]["KP"]["unit"]
            self.pos_amp.ki_unit = k["position_loop"]["KI"]["unit"]
            self.pos_amp.kd_unit = k["position_loop"]["KD"]["unit"]

            self.vel_amp.kp_unit = k["velocity_loop"]["KP"]["unit"]
            self.vel_amp.ki_unit = k["velocity_loop"]["KI"]["unit"]
            self.vel_amp.kd_unit = k["velocity_loop"]["KD"]["unit"]

            # Command limitations:
            
            self.vel_cmd_lim_val = k["position_loop"]["vel_Cmd_limitation"]["value"]
            self.tor_cmd_lim_val = k["velocity_loop"]["tor_Cmd_limitation"]["value"]
            self.vel_cmd_lim.unit = k["position_loop"]["vel_Cmd_limitation"]["unit"]
            self.tor_cmd_lim.unit = k["velocity_loop"]["tor_Cmd_limitation"]["unit"]

            self.vel_cmd_lim.setLimitation((-1*self.vel_cmd_lim_val, self.vel_cmd_lim_val))
            
            self.tor_cmd_lim.setLimitation((-1*self.tor_cmd_lim_val, self.tor_cmd_lim_val))

            self.vel_cmd_filter.setup(k["position_loop"]["filter"]["Time_constant"]["value"])
            self.vel_cmd_filter.unit = k["position_loop"]["filter"]["Time_constant"]["unit"]

            self.tor_cmd_filter.unit = k["current_loop"]["Cutoff_frequency"]["unit"]
            self.tor_cmd_filter.Setup(FilterType.fir, k["current_loop"]["order"], k["current_loop"]["Cutoff_frequency"]["value"])
            
            #           
    #
    def setInitial(self, pos_init:np.float32=0.0, vel_init:np.float32=0.0, acc_init:np.float32=0.0, tor_init:np.float32=0.0):
        self.__pos = pos_init
        self.__vel = vel_init
        self.__acc = acc_init
        self.__tor = tor_init
        self.__pos_internal = self.pos_unit_to_internal(pos_init)
        self.__vel_internal = self.vel_unit_to_internal(vel_init)
        self.__acc_internal = self.acc_unit_to_internal(acc_init)
        self.__tor_internal = self.tor_unit_to_internal(tor_init)
        self.motor.setInit(theta=pos_init)
    #
    def setPID(self, gain:any, value:np.float32):
        '''
        example: (tune position-loop Kp gain 20.0)
            j1 = JointServoDrive(id = 1)
            j1.TuneGain(ServoGain.Position.kp, 20.0)
        '''
        if gain == ServoGain.Position.value.kp:
            self.pos_amp.kp = value
        elif gain == ServoGain.Position.value.ki:
            self.pos_amp.ki = value
        elif gain == ServoGain.Velocity.value.kp:
            self.vel_amp.kp = value
        elif gain == ServoGain.Velocity.value.ki:
            self.vel_amp.ki = value
    #
    def setMotorModel(self, item:MotorModel, value:np.float32):
        '''
        example: (modify Jm of motor 0.1)
        '''
        if item == MotorModel.Jm:
            self.motor.Jm = value
        elif item == MotorModel.fric_vis:
            self.motor.fric_vis = value
        elif item == MotorModel.fric_c:
            self.motor.fric_Coulomb = value
        elif item == MotorModel.fric_dv:
            self.motor.fric_dv = value
    #
    def EnableNonlinearEffect(self, en:bool):
            self.__nonlinear_enabled = en
    #
    def EnableCommandFilter(self, en:bool=False):
        self.en_cmd_filters = en
    #
    def NonlinearEnabled(self):
        return self.__nonlinear_enabled
    #
    def __call__(self, xr:np.float32, tor_n:np.float32=0.0):
    
        '''
         xr: input
         tor_n: nonlinear coupling effect
        '''
        xr_internal = self.xr_unit_to_internal(x=xr)
        # --------------- Position Loop: ---------------
        self.pos_err_internal = self.pos_node(xr_internal, -1.0*self.__pos_internal)
        # if self.__joint_ID==1:
        #     print(self.pos_err_internal)
        self.vel_cmd_internal = self.pos_amp(self.pos_err_internal )
        # if self.__joint_ID==1:
        #     print(self.vel_cmd_internal)
        #     print(f'vel_cmd_internal:{self.vel_cmd_internal}')
        self.vel_cmd_internal = self.vel_cmd_lim(self.vel_cmd_internal)
        if self.en_cmd_filters:
            self.vel_cmdf_internal = self.vel_cmd_filter(self.vel_cmd_internal)
        else:
            self.vel_cmdf_internal = self.vel_cmd_internal

        # if self.__joint_ID==1:
        #     print(self.vel_cmdf_internal)
        #     print(f'vel_cmdf_internal:{self.vel_cmdf_internal}')
        # --------------- Velocity Loop: ---------------
        self.vel_err_internal = self.vel_node(self.vel_cmdf_internal, -1.0*self.__vel_internal)
        self.tor_cmd_internal = self.vel_amp(self.vel_err_internal)
        self.tor_cmd_internal = self.tor_cmd_lim(self.tor_cmd_internal)

        # if self.__joint_ID==1:
        #     print(f'tor_cmd_internal:{self.tor_cmd_internal}')
        if self.en_cmd_filters:
            self.tor_cmdf_internal = self.tor_cmd_filter(self.tor_cmd_internal)
        else:
            self.tor_cmdf_internal = self.tor_cmd_internal
        # if self.__joint_ID==1:
        #     print(f'tor_cmdf_internal:{self.tor_cmdf_internal}')
            
        # --------------- Torque Loop: ---------------
        self.tor_cmdf_internal = self.g_tor_node(self.tor_cmdf_internal, self.tor_unit_to_internal(tor_n))

        # if self.__joint_ID==1:
        #     print(f'tor_n:{tor_n}')
        # if self.__joint_ID==1:
        #     print(f'tor_cmdf_internal:{self.tor_cmdf_internal}')
        self.__tor_internal = self.tor_amp(self.tor_cmdf_internal) #Ratio=1

        # if self.__joint_ID==1:
        #     print(f'tor_internal:{self.__tor_internal}')

        #Torque real output unit ==> real value(外部看到的)
        self.__tor = self.tor_unit_to_internal(self.__tor_internal, inverse=True) # unit:Ｎｍ
        # Motor Model:
        self.__acc, self.__vel, self.__pos = \
            self.motor(ctrl_mode="torque", u=self.__tor)
        # print(self.__tor.shape)
        
        # unit transformations: (內部資訊==>供之後參考)
        self.__pos_internal = self.pos_unit_to_internal(self.__pos)
        self.__vel_internal = self.vel_unit_to_internal(self.__vel)
        self.__acc_internal = self.acc_unit_to_internal(self.__acc)
        # translate to joint space:
        q   = self.reducer(self.__pos)
        dq  = self.reducer(self.__vel)
        ddq = self.reducer(self.__acc)
        
        
        # print(f'{self.__joint_ID}: {self.__pos}')
        return self.__pos, dq, ddq, self.__tor, self.pos_unit_to_internal(self.pos_err_internal, inverse=True), self.vel_unit_to_internal(self.vel_cmd_internal, inverse=True)
    #
    
    def export_servo_gain(self,gain_settin_file_name:str):
        with open(self.model_path) as fs:
            drv_info = json.loads(fs.read())
            model_save_path = drv_info["model_save_path"]

        data = {
            "position_loop": {
                "KP": {"unit": "1/second", "value": self.pos_amp.kp},
                "KI": {"unit": "0.1-ms", "value": self.pos_amp.ki},
                "KD": {"unit": "0.1-ms", "value": self.pos_amp.kd},
                "vel_Cmd_limitation": {"unit": "0.1-ms", "value": self.vel_cmd_lim_val},
                "filter": {
                    "filter_type": "exponential_decay",
                    "Time_constant": {"unit": "s", "value": self.vel_cmd_filter.time_const}
                }
            },
            "velocity_loop": {
                "KP": {"unit": "Hz", "value": self.vel_amp.kp},
                "KI": {"unit": "0.1-ms", "value": self.vel_amp.ki},
                "KD": {"unit": "0.1-ms", "value": self.vel_amp.kd},
                "tor_Cmd_limitation": {"unit": "0.1-ms", "value": self.tor_cmd_lim_val}
            },
            "current_loop": {
                "filter_type": "fir",
                "order": self.tor_cmd_filter.get_order(),
                "Cutoff_frequency": {"unit": "Hz", "value": self.tor_cmd_filter.get_fc()}
            }
        }

        # Writing JSON data to a file
        with open(self.model_path + model_save_path + gain_settin_file_name, "w") as json_file:
            json.dump(data, json_file, indent=4)

        print("JSON data has been written to 'output.json'")
'''
J1 = JointServoDrive(id=1, saved_model="./j1.sdrv")
J1.setInitial(0.0, 0.0, 0.0, 0.0)
q, dq, ddq, tor, pos_err = J1(xr=0.001, 0.0)
'''