#%%
from libs.RobotManipulators import RT605_710
import libs.ServoDriver as dev
import math
import numpy as np
from threading import Thread
from libs.ServoMotor import ServoMotor
from libs.type_define import *
from libs import ControlSystem as cs
from libs import ServoDriver
from libs.ForwardKinematic import FowardKinematic
from libs.rt605_Gtorq_model import RT605_GTorq_Model
import json
import time
import csv
from scipy.fft import fft, fftfreq, ifft

### plot library
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.basemap import Basemap

from libs.bode_plot import Freq_Response



#%%
#####################################
#                                   #
#      Read Interpolation Path      #
#                                   #
#####################################

path_file_dir = './data/Path/'
path_name = 'XY_circle_path.txt'

data = np.genfromtxt(path_file_dir+path_name, delimiter=',')
# deterning if it is XY circular test or YZ circular test or line test
if path_name.find("XY") !=-1:
    path_mode = 0
elif path_name.find("YZ") != -1:
    path_mode = 1
elif path_name.find("line") != -1:
    path_mode = 2
# Sampling rate
ts = 0.001 

X_c = data[:,0]/1000000
Y_c = data[:,1]/1000000
Z_c = data[:,2]/1000000

pitch_c = data[:, 3]/1000
yaw_c = data[:, 4]/1000
roll_c = data[:,5]/1000

q_c = np.zeros((2885, 6))


q1_c = data[:,6]/1000
q2_c = data[:,7]/1000
q3_c = data[:,8]/1000
q4_c = data[:,9]/1000
q5_c = data[:,10]/1000
q6_c = data[:,11]/1000

# Concatenate the arrays into a single 2-dimensional array
q_c = np.column_stack((q1_c,q2_c,q3_c,q4_c,q5_c,q6_c))
ori_c = np.column_stack((pitch_c, yaw_c, roll_c))


#%%
#####################################
#                                   #
#      Construct 6 Servo motor      #
#                                   #
#####################################

model_path = './data/servos/'
log_path = './data/log/'

joints:ServoDriver.JointServoDrive = [None]*6


for i in range(6):
    model_path_name = f"j{i+1}/j{i+1}.sdrv"
    print(model_path+model_path_name)
    joints[i] = ServoDriver.JointServoDrive(id=i,saved_model=model_path+model_path_name)
    joints[i].setInitial()




forward_kinematic = FowardKinematic(unit='degree')


arr_size = q1_c.shape[0]
q = np.zeros((arr_size, 6))
t = ts * np.arange(0,arr_size)

x = np.zeros(arr_size)
y = np.zeros(arr_size)
z = np.zeros(arr_size)
x_c = np.zeros(arr_size)
y_c = np.zeros(arr_size)
z_c = np.zeros(arr_size)
pitch = np.zeros(arr_size)
roll = np.zeros(arr_size)
yaw = np.zeros(arr_size)

q_pos_err = np.zeros((arr_size,6))
torque = np.zeros((arr_size,6))

# Compute gravity torque
compute_GTorque = RT605_GTorq_Model()
g_tor = np.zeros(6,dtype=np.float32)



#%%
for i, q_ref in enumerate(zip(q1_c,q2_c,q3_c,q4_c,q5_c,q6_c)):
    for idx in range(6):

        pos,vel,acc,tor,pos_err,vel_err = joints[idx](q_ref[idx]-q_c[0][idx],g_tor[idx])
        q[i][idx] = pos+q_c[0][idx]
        q_pos_err[i][idx] = pos_err
        torque[i][idx] = tor

    g_tor = compute_GTorque(q[i][1],q[i][2],q[i][3],q[i][4],q[i][5])

    x[i],y[i],z[i],pitch[i],roll[i],yaw[i] = forward_kinematic((q[i,0],q[i,1],q[i,2],q[i,3],q[i,4],q[i,5]))
    #x_c[i],y_c[i],z_c[i], = forward_kinematic((q1_c[i],q2_c[i],q3_c[i],q4_c[i],q5_c[i],q6_c[i]))

        # pos,vel,acc,tor,pos_err,vel_err = motor2(q2-q2_c[0],g_tor[1])
        # q[i][1] = pos+q2_c[0]
        # q_pos_err[i][1] = pos_err
        # torque[i][1] = tor

        # pos,vel,acc,tor,pos_err,vel_err = motor3(q3-q3_c[0],g_tor[2])
        # q[i][2] = pos+q3_c[0]
        # q_pos_err[i][2] = pos_err
        # torque[i][2] = tor

        # pos,vel,acc,tor,pos_err,vel_err = motor4(q4-q4_c[0],g_tor[3])
        # q[i][3] = pos+q4_c[0]
        # q_pos_err[i][3] = pos_err
        # torque[i][3] = tor

        # pos,vel,acc,tor,pos_err,vel_err = motor5(q5-q5_c[0],g_tor[4])
        # q[i][4] = pos+q5_c[0]
        # q_pos_err[i][4] = pos_err
        # torque[i][4] = tor

        # pos,vel,acc,tor,pos_err,vel_err = motor6(q6-q6_c[0],g_tor[5])
        # q[i][5] = pos+q6_c[0]
        # q_pos_err[i][5] = pos_err
        # torque[i][4] = tor

    
    

    # q[i][0],q_pos_err[i][0] = motor1(q1-q1_c[0])+q1_c[0]
    # q[i][1],q_pos_err[i][1] = motor2(q2-q2_c[0])+q2_c[0]
    # q[i][2],q_pos_err[i][2] = motor3(q3-q3_c[0])+q3_c[0]
    # q[i][3],q_pos_err[i][3] = motor4(q4-q4_c[0])+q4_c[0]
    # q[i][4],q_pos_err[i][4] = motor5(q5-q5_c[0])+q5_c[0]
    # q[i][5],q_pos_err[i][5] = motor6(q6-q6_c[0])+q6_c[0]
    
    
    

### System log ###
# np.savetxt(log_path+'pos_error.txt',q_pos_err,delimiter=',',header='Joint1, Joint2, Joint3, Joint4, Joint5, Joint6', fmt='%10f')
# np.savetxt(log_path+'pos.txt',q,delimiter=',',header='Joint1, Joint2, Joint3, Joint4, Joint5, Joint6', fmt='%10f')
# np.savetxt(log_path+'tor.txt',torque,delimiter=',',header='Joint1, Joint2, Joint3, Joint4, Joint5, Joint6', fmt='%10f')

#%%
### plot frequency response of motor system

motor_freq_response = Freq_Response()
motors = [joints[0], joints[1], joints[2], joints[3], joints[4], joints[5]]
motor_freq_response(motors)


#%%
### plot the result ###
t = np.array(range(0,arr_size))*ts
fig,ax = plt.subplots(3,2)

# Set the same scale for each axis
max_range = np.array([q[:,0].max()-q[:,0].min(), 
                      q[:,1].max()-q[:,1].min(),
                      q[:,2].max()-q[:,2].min(),
                      q[:,3].max()-q[:,3].min(),
                      q[:,4].max()-q[:,4].min(),
                      q[:,5].max()-q[:,5].min()]).max() / 2.0
mid_q1 = (q[:,0].max()+q[:,0].min()) * 0.5 
mid_q2 = (q[:,1].max()+q[:,1].min()) * 0.5 
mid_q3 = (q[:,2].max()+q[:,2].min()) * 0.5
mid_q4 = (q[:,3].max()+q[:,3].min()) * 0.5 
mid_q5 = (q[:,4].max()+q[:,4].min()) * 0.5 
mid_q6 = (q[:,5].max()+q[:,5].min()) * 0.5

mid_q = (mid_q1,mid_q2,mid_q3,mid_q4,mid_q5,mid_q6)


for i in range(6):
    
    ax[i//2,i%2].set_title(f"joint{i+1}")
    ax[i//2,i%2].plot(t,q[:,i],label='actual')
    ax[i//2,i%2].plot(t,q_c[:,i],label='ref')
    ax[i//2,i%2].grid(True)
    ax[i//2,i%2].set_ylim(mid_q[i] - 1.1 * max_range, mid_q[i] + 1.1 * max_range)
    ax[i//2,i%2].set_xlabel("time(s)")
    ax[i//2,i%2].set_ylabel(r"$\theta$(deg)")
    ax[i//2,i%2].legend(loc='best')


plt.suptitle('Joint angle')
plt.tight_layout()


fig,ax = plt.subplots(3,2)

# Set the same scale for each axis
max_range = np.array([q_pos_err[:,0].max()-q_pos_err[:,0].min(), 
                      q_pos_err[:,1].max()-q_pos_err[:,1].min(),
                      q_pos_err[:,2].max()-q_pos_err[:,2].min(),
                      q_pos_err[:,3].max()-q_pos_err[:,3].min(),
                      q_pos_err[:,4].max()-q_pos_err[:,4].min(),
                      q_pos_err[:,5].max()-q_pos_err[:,5].min()]).max() / 2.0
mid_q1_err = (q_pos_err[:,0].max()+q_pos_err[:,0].min()) * 0.5 
mid_q2_err = (q_pos_err[:,1].max()+q_pos_err[:,1].min()) * 0.5 
mid_q3_err = (q_pos_err[:,2].max()+q_pos_err[:,2].min()) * 0.5
mid_q4_err = (q_pos_err[:,3].max()+q_pos_err[:,3].min()) * 0.5 
mid_q5_err = (q_pos_err[:,4].max()+q_pos_err[:,4].min()) * 0.5 
mid_q6_err = (q_pos_err[:,5].max()+q_pos_err[:,5].min()) * 0.5

mod_q_err = (mid_q1_err,mid_q2_err,mid_q3_err,mid_q4_err,mid_q5_err,mid_q6_err)

for i in range(6):
    ax[i//2,i%2].set_title(f"joint{i+1}")
    ax[i//2,i%2].plot(t,q_pos_err[:,i])
    ax[i//2,i%2].grid(True)
    ax[i//2,i%2].set_ylim(mod_q_err[i] - 1.1 * max_range, mod_q_err[i]  + 1.1 * max_range)
    ax[i//2,i%2].set_xlabel("time(s)")
    ax[i//2,i%2].set_ylabel(r"$\theta$(deg)")



plt.suptitle('Joint angle error')
plt.tight_layout()

# create x,y,z plot

# Set the same scale for each axis
max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0 * 1000
mid_x = (x.max()+x.min()) * 0.5 * 1000
mid_y = (y.max()+y.min()) * 0.5 * 1000
mid_z = (z.max()+z.min()) * 0.5 * 1000

fig,ax = plt.subplots(3,1)
ax[0].plot(t,x*1000, label='actual')
ax[0].plot(t,X_c*1000, label='ref')
ax[0].set_xlabel('time(s)')
ax[0].set_ylabel('X(mm)')
ax[0].set_ylim(mid_x - 1.1 * max_range, mid_x + 1.1 * max_range)
ax[0].grid(True)
ax[0].legend(loc='best')

ax[1].plot(t,y*1000)
ax[1].plot(t,Y_c*1000)
ax[1].set_xlabel('time(s)')
ax[1].set_ylabel('Y(mm)')
ax[1].set_ylim(mid_y - 1.1 * max_range, mid_y + 1.1 * max_range)
ax[1].grid(True)


ax[2].plot(t,z*1000)
ax[2].plot(t,Z_c*1000)
ax[2].set_xlabel('time(s)')
ax[2].set_ylabel('Z(mm)')
ax[2].set_ylim(mid_z - 1.1 * max_range, mid_z  + 1.1 * max_range)
ax[2].grid(True)



# Create 3D plot
fig = plt.figure()
ax = plt.axes(projection='3d')



# Add data to plot
# ax.scatter(X_c, Y_c, Z_c, s=1)
ax.scatter(X_c*1000, Y_c*1000, Z_c*1000, s=1,label="ref")
ax.scatter(x*1000,y*1000,z*1000,s=1,label="actual")
ax.scatter(X_c[0]*000,Y_c[0]*1000,Z_c[0]*1000, c='red',marker='*',s=100)
ax.text(X_c[0]*000,Y_c[0]*1000,Z_c[0]*1000,"start",color='red')
ax.legend(loc='best')

# Set the same scale for each axis
max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0 * 1000
mid_x = (x.max()+x.min()) * 0.5 * 1000
mid_y = (y.max()+y.min()) * 0.5 * 1000
mid_z = (z.max()+z.min()) * 0.5 * 1000
ax.set_xlim(mid_x -1.1 *  max_range, mid_x +1.1 *  max_range)
ax.set_ylim(mid_y -1.1 *  max_range, mid_y +1.1 *  max_range)
ax.set_zlim(mid_z -1.1 *  max_range, mid_z +1.1 *  max_range)


# Set labels and title
ax.set_xlabel('X(mm)')
ax.set_ylabel('Y(mm)')
ax.set_zlabel('Z(mm)')
ax.set_title('3D XYZ plot')

# Show the plot
plt.show()



# %%

#　This function plot the circular trajectory tracking error in polar form


# determine the center of circle
x_offset = (max(x_c) + min(x_c))/2 * 1000
y_offset = (max(y_c) + min(y_c))/2 * 1000
z_offset = (max(z_c) + min(z_c))/2 * 1000

print(f"X center {x_offset}")
print(f"Y offset {y_offset}")
print(f"Z offset {z_offset}")


r = np.zeros(arr_size)
phi = t

r_c = np.zeros(arr_size)
phi_c = np.zeros(arr_size)



rho = np.zeros(arr_size)
print(t[-1])
if path_mode == 0: # XY circular test

    for i in range(arr_size):
        r[i] = math.sqrt((x[i]*1000 - x_offset)**2 + (y[i]*1000  - y_offset)**2)
        phi[i] = t[i]/t[-1] * 2* math.pi

        r_c[i] = math.sqrt((x_c[i]*1000  - x_offset)**2 + (y_c[i]*1000  - y_offset)**2)
        phi_c[i] = t[i]/t[-1] * 2* math.pi

        rho[i] = r_c[i] - r[i]



elif path_mode == 1: # YZ circular test
    for i in range(arr_size):
        r[i] = math.sqrt((y[i]*1000  - y_offset)**2 + (z[i]*1000  - z_offset)**2)
        phi[i] =  t[i]/t[-1] * 2* math.pi

        r_c[i] = math.sqrt((y_c[i]*1000  - y_offset)**2 + (z_c[i]*1000  - z_offset)**2)
        phi_c[i] = t[i]/t[-1] * 2* math.pi

        rho[i] = r_c[i] - r[i]

# print(phi)

radius_range = max(rho) - min(rho)

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(phi, rho)
ax.set_rmax(max(rho) + 0.2*radius_range)
ax.set_rmin(min(rho) - 0.2*radius_range)
# ax.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
ax.grid(True)

plt.show()

# %%
