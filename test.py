from rt605 import RT605
import numpy as np


rt605 = RT605(ts=0.0005)

command = np.array([0.1,0.1,0.1,0.1,0.1,0.1])
x,y,z,pitch, row, yaw = rt605(command)

print(x,y,z,pitch,row,yaw)
