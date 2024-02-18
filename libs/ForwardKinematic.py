from libs.type_define import*
import numpy as np
from scipy import signal
import math


class FowardKinematic():
    def __init__(self,unit:str,offset=[0,0,120]) -> None:
        self.unit = unit
        self.offset = offset
        
        
    def __call__(self, joint:np.array):


        if(self.unit=='degree'):
            self.q1 = np.deg2rad(joint[0])
            self.q2 = np.deg2rad(joint[1])
            self.q3 = np.deg2rad(joint[2])
            self.q4 = np.deg2rad(joint[3])
            self.q5 = np.deg2rad(joint[4])
            self.q6 = np.deg2rad(joint[5])
        else:
            self.q1 =  joint[0]
            self.q2 =  joint[1]
            self.q3 =  joint[2]
            self.q4 =  joint[3]
            self.q5 =  joint[4]
            self.q6 =  joint[5]

        T = np.array(self.offset, ndmin=2).T   #bias term (tool frame)

        t2 = np.cos(self.q1)
        t3 = np.cos(self.q2)
        t4 = np.cos(self.q3)
        t5 = np.cos(self.q4)
        t6 = np.cos(self.q5)
        t7 = np.cos(self.q6)
        t8 = np.sin(self.q1)
        t9 = np.sin(self.q2)
        t10 = np.sin(self.q3)
        t11 = np.sin(self.q4)
        t12 = np.sin(self.q5)
        t13 = np.sin(self.q6)
        t14 = t3*t4
        t15 = t3*t10
        t16 = t4*t9
        t17 = t9*t10
        t18 = t8*t17
        t19 = -t17
        t20 = t2*t14
        t21 = t8*t14
        t22 = t2*t17
        t25 = t15+t16
        t23 = -t21
        t24 = t2*t19
        t26 = t14+t19
        t27 = t5*t6*t25
        t28 = t11*t13*t25
        t29 = t12*t26
        t30 = t20+t24
        t31 = t18+t23
        t32 = -t28
        t33 = t28*1j
        t34 = t27+t29
        t35 = t7*t34
        t36 = t35*1j
        t38 = t32+t35
        t39 = (t28-t35)**2
        t37 = -t36
        t40 = -t39
        t41 = t40+1.0
        t42 = np.sqrt(t41)

        ori = np.array(np.rad2deg([np.arctan2(-t13*t34-t7*t11*t25,-t6*t26+t5*t12*t25),
                        np.angle(t33+t37+t42),
                        np.angle(t33+t37-t42),
                        np.arctan2(-t7*(t6*(t2*t11+t5*t31)+t12*(t8*t15+t8*t16))-t13*(t2*t5-t11*t31),-t7*(t12*(t2*t15+t2*t16)-t6*(t8*t11+t5*t30))+t13*(t5*t8-t11*t30))]))
        
        for i in range(len(ori)):
            if ori[i]== -180:
                ori[i]+=360

        pitch = ori[0]
        roll = ori[1]
        yaw = ori[3]


        v2 =  np.cos(self.q1) 
        v3 =  np.sin(self.q1) 
        v4 =  np.sin(self.q4) 
        v5 =  np.cos(self.q4) 
        v6 =  np.cos(self.q2) 
        v7 =  np.cos(self.q3) 
        v8 =  np.sin(self.q2) 
        v9 =  np.sin(self.q3) 
        v13 = v2*v6*v7 
        v14 = v2*v8*v9 
        v10 = v13-v14 
        v11 =  np.cos(self.q6) 
        v12 = v3*v5 
        v15 = v12-v4*v10 
        v16 =  np.sin(self.q6) 
        v17 =  np.cos(self.q5) 
        v18 = v3*v4 
        v19 = v5*v10 
        v20 = v18+v19 
        v21 =  np.sin(self.q5) 
        v22 = v2*v6*v9 
        v23 = v2*v7*v8 
        v24 = v22+v23 
        v25 = v17*v20-v21*v24 
        v26 = v3*v6*v7 
        v29 = v3*v8*v9 
        v27 = v26-v29 
        v28 = v2*v5 
        v30 = v4*v27 
        v31 = v28+v30 
        v32 = v2*v4 
        v40 = v5*v27 
        v33 = v32-v40 
        v34 = v17*v33 
        v35 = v3*v6*v9 
        v36 = v3*v7*v8 
        v37 = v35+v36 
        v38 = v21*v37 
        v39 = v34+v38 
        v41 = self.q2+self.q3 
        v42 =  np.sin(v41) 
        v43 =  np.cos(v41) 
        v44 = v21*v43 
        v45 = v5*v17*v42 
        v46 = v44+v45 

        T07 = np.array([v15*v16+v11*v25,
                       -v16*v31-v11*v39,
                       v11*v46-v4*v16*v42,
                       0.0,
                       v11*v15-v16*v25,
                       -v11*v31+v16*v39,
                       -v16*v46-v4*v11*v42,
                       0.0,
                       v17*v24+v20*v21,
                       v17*v37-v21*v33,
                       -v17*v43+v5*v21*v42,
                       0.0,
                       v2*3.0e1+v2*v6*3.4e2+v2*v6*v7*4.0e1+v2*v6*v9*3.38e2+v2*v7*v8*3.38e2-v2*v8*v9*4.0e1+v3*v4*v21*(1.73e2/2.0)+v2*v17*v42*(1.73e2/2.0)+v2*v5*v6*v7*v21*(1.73e2/2.0)-v2*v5*v8*v9*v21*(1.73e2/2.0),
                       v3*3.0e1+v3*v6*3.4e2+v3*v6*v7*4.0e1+v3*v6*v9*3.38e2+v3*v7*v8*3.38e2-v3*v8*v9*4.0e1-v2*v4*v21*(1.73e2/2.0)+v3*v17*v42*(1.73e2/2.0)+v3*v5*v6*v7*v21*(1.73e2/2.0)-v3*v5*v8*v9*v21*(1.73e2/2.0),
                       v8*3.4e2+v42*4.0e1-v43*3.38e2-v17*v43*(1.73e2/2.0)+v42* np.sin(self.q4+self.q5)*(1.73e2/4.0)-v42* np.sin(self.q4-self.q5)*(1.73e2/4.0)
                       ,1.0])

        T07 = np.reshape(T07,(4,4),order='F')
        #T07 = T07*[np.eye(3), T [0, 0, 0, 1]] 
        #print(T07)

        # Define a 4x4 identity matrix
        I = np.identity(3)
        T_eye = np.hstack((I, T))

        # Create a 3x4 matrix by horizontally stacking the 3x3 identity matrix and T with a row of zeros
        T_eye = np.hstack((np.identity(3), T))
        T_eye = np.vstack((T_eye, np.array([0, 0, 0, 1])))

        # Multiply T07 by the 4x4 matrix formed from T_eye
        T07 = T07.dot(T_eye)
        #print(T07)
        
        #T07[0, 3] = T07[0, 3]/1000

        X = T07[0,3]/1000
        Y = T07[1,3]/1000
        Z = T07[2,3]/1000

        return X,Y,Z,pitch,roll,yaw