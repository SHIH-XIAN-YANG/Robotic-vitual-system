from rt605 import RT605
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np

# one point mode 
# rt605 is consider as a box and input q_c is desired joint trajectory...
# x,y,z,pitch, roll yaw is the coresponding output
def main():

    # initialize RT605
    rt605 = RT605()
    # rt605.initialize_model() # load servo inforamtion

    #load path
    path_file_dir = './data/Path/'
    path_name = "sine_f6_full_joints.txt"
    
    # load_HRSS_trajectory --> Return joint trajectory in column vector (ndarray)
    q_c = rt605.load_HRSS_trajectory(path_file_dir+path_name) 
    rt605.forward_kinematic.setOffset([0,0,0])
    
    # Personalize rt605 (set gain or enable nonlinear effect)
    # for example: set PID (Kpp) gain  
    # rt605.setPID(i,gain="kpp",value=100)

    # example enable gravity effect
    rt605.compute_GTorque.enable_Gtorq(en=True)
    rt605.compute_friction.enable_friction(en=True)
    

    ts = 0.0005
    time = []
    x = []
    y = []
    z = []
    pitch = []
    roll = []
    yaw = []

    rt605.run_HRSS_intp()
    # rt605.plot_cartesian()

    # # forward
    # for i,q_ref in enumerate(q_c):

    #     x_t,y_t,z_t,pitch_t,roll_t,yaw_t = rt605(q_ref)

        
    #     time.append(i*ts)
    #     x.append(x_t)
    #     y.append(y_t)
    #     z.append(z_t)
    #     pitch.append(pitch_t)
    #     roll.append(roll_t)
    #     yaw.append(yaw_t)

    gain_deviation = np.zeros(6)

    for idx in range(6):
        # print(max(rt605.q_c[:,idx]), max(rt605.q[:, idx]))
        gain_deviation[idx] = max(rt605.q_c[:,idx]) - max(rt605.q[:, idx])
    
    print(gain_deviation)

    phase_deviation = np.zeros(6)
    for idx in range(6):
        # print(find_closest_index(rt605.q[:, idx], rt605.q_c[0, idx]), find_closest_index(rt605.q_c[:, idx], rt605.q_c[0, idx]))
        phase_deviation[idx] = rt605.ts * (find_closest_index(rt605.q[:, idx], rt605.q_c[0, idx]) - find_closest_index(rt605.q_c[:, idx], rt605.q_c[0, idx]))
    print(phase_deviation)

    # show results
    # plt.plot(time,x,time,y,time,z)
    # plt.show()
    
    rt605.plot_joint()

    # rt605.plot_cartesian()

    rt605.plot_error()
    # rt605.plot_polar()
    # rt605.sweep()

def compute_gain_error():
    return 

def find_closest_index(data, target):
    min_diff = float('inf')
    closest_index = -1

    quarter_len = len(data) // 4
    three_quarter_len = (len(data)*3) // 4

    for idx in range(quarter_len, three_quarter_len):
        value = data[idx]
        diff = abs(value - target)

        if diff<min_diff:
            min_diff = diff
            closest_index = idx
    return closest_index

if __name__ == "__main__":
    main()