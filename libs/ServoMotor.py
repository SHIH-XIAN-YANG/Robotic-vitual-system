#
import numpy as np
import matplotlib.pyplot as plt
import json
# 
class ServoMotor:
    __id:int = None
    def __init__(self, id:int):
        self.setID(id)
        # -------------------------------------------------------------
        # Vendor infomations: (identical to the CiA402 object dictionary)
        # -------------------------------------------------------------
        self.info_manufacturer = str()
        self.info_model = str()
        self.info_motor_type = str() # either DC or AC 
        # -------------------------------------------------------------
        # Electrical parameters: 馬達廠商資料手冊提供之固定的電氣參數
        # -------------------------------------------------------------
        self.kt = np.float32()      # unit:
        self.kb = np.float32()      # unit: 
        self.Ra = np.float32()      # unit: ohm
        self.La = np.float32()      # unit: mH
        # -------------------------------------------------------------
        # Dynamics parameters: 馬達廠商資料手冊提供之固定的動力學參數
        # -------------------------------------------------------------
        self.Jm = np.float32() # 2nd moment inertia of the mass
        self.fric_vis = np.float32() # friction viscosity
        self.Jm_unit:str = None
        self.fric_vis_unit:str = None
        # Non-linear friction terms:
        self.fric_Coulomb = np.float32()
        self.fric_dv = np.float32()
        # -------------------------------------------------------------
        # 模擬離散時域之狀態用：
        # -------------------------------------------------------------
        self.ts = np.float32() # discrete-time step or sampling time
        # 馬達運動狀態：
        self.theta = np.float32()    # unit: rad
        self.D_theta = np.float32()  # unit: rad*second^-1
        self.DD_theta = np.float32() # unit: rad*second^-2
        
        self.torque = np.float32()   # unit: N/m
        self.Ia = np.float32()       # unit: mA
    #

    def setInit(self, theta: np.float32):
        self.theta = theta
        
    def setID(self, _id:int):
        self.__id = _id
    #
    def ID(self):
        return self._id
    #
    def ImportMotorModel(self, saved_model:str):
        '''
        讀取外部 JSON 或 ESI 之 XML file。
        '''
        if saved_model[-4:] != ".mot":
            print("Error syntax.")
            return False
        else:
            with open(saved_model) as fs:
                mot_info = json.loads(fs.read())
                # -------------------------------------------------------------
                # Vendor infomations: (identical to the CiA402 object dictionary)
                # -------------------------------------------------------------
                self.info_manufacturer = mot_info["manufacturer"]
                self.info_model = mot_info["model"]
                self.info_motor_type = mot_info["power_supply"]["type"] 
                # -------------------------------------------------------------
                # Electrical parameters: 馬達廠商資料手冊提供之固定的電氣參數
                # -------------------------------------------------------------
                self.kt = mot_info["kt"]["value"]
                self.kb = mot_info["kb"]["value"] 
                self.Ra = mot_info["Ra"]["value"]
                self.La = mot_info["La"]["value"]
                # -------------------------------------------------------------
                # Dynamics parameters: 馬達廠商資料手冊提供之固定的動力學參數
                # -------------------------------------------------------------
                self.Jm = mot_info["Jm"]["value"]
                self.fric_vis = mot_info["friction"]["viscosity"]["value"]
                self.Jm_unit = mot_info["Jm"]["unit"]
                self.fric_vis_unit = mot_info["friction"]["viscosity"]["unit"]
                
                # Non-linear friction terms:
                self.fric_Coulomb = mot_info["friction"]["Coulomb"]["value"]
                self.fric_dv = mot_info["friction"]["velocity_power"]["value"]
                # -------------------------------------------------------------
                # 模擬離散時域之狀態用：
                # -------------------------------------------------------------
                self.ts = mot_info["ts"]["value"]
                
    #
    def Setup(self, ts:np.float32, Jm:np.float32):
        pass
    #
    def ShowParameter(self):
        print(f"\Motor manufacturer: {self.info_manufacturer}\n\
                Model: {self.info_model}\n\
                Motor type: {self.info_motor_type}")
    #
    def NonlinearFric(self):
        return 1
    # overloading operator ()
    def __call__(self, u, ctrl_mode="torque"):
        '''
           [I/P]:
                - ctrl_mode: 馬達輸入的控制訊號，可以是 "torque" 或是 "current"。
                - ret: 回傳計算結果，分別有 "pos", "vel", "acc" 或 "acc-vel-pos"。
        '''
        t1 = self.DD_theta
        t2 = self.D_theta
        if ctrl_mode == "current":
            self.Ia = u
            self.torque = self.kt*self.Ia
            self.DD_theta = (self.torque - self.fric_vis*self.D_theta)/self.Jm
            self.D_theta += (t1 + self.DD_theta)*self.ts/2
            self.theta += (t2 + self.D_theta)*self.ts/2
        elif ctrl_mode == "torque":
            self.torque = u
            self.DD_theta = (self.torque - self.fric_vis*self.D_theta)/self.Jm # u = Jsy+by
            self.D_theta += (t1 + self.DD_theta)*self.ts/2
            self.theta += (t2 + self.D_theta)*self.ts/2
        
        return self.DD_theta, self.D_theta, self.theta
    #

        