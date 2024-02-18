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
    rt605.initialize_model() # load servo inforamtion

    #load path
    path_file_dir = './data/Path/'
    path_name = "XY_circle_path.txt"
    
    # load_HRSS_trajectory --> Return joint trajectory in column vector (ndarray)
    q_c = rt605.load_HRSS_trajectory(path_file_dir+path_name) 
    
    

    # Personalize rt605 (set gain or enable nonlinear effect)
    # for example: set PID (Kpp) gain  
    # rt605.setPID(i,gain="kpp",value=100)

    # example enable gravity effect
    rt605.compute_GTorque.enable_Gtorq(en=True)
    rt605.compute_friction.enable_friction(en=True)

    ts = 0.001
    time = []
    x = []
    y = []
    z = []
    pitch = []
    roll = []
    yaw = []

    rt605.start()
    # rt605.plot_cartesian()

    # forward
    for i,q_ref in enumerate(q_c):

        x_t,y_t,z_t,pitch_t,roll_t,yaw_t = rt605(q_ref)

        
        time.append(i*ts)
        x.append(x_t)
        y.append(y_t)
        z.append(z_t)
        pitch.append(pitch_t)
        roll.append(roll_t)
        yaw.append(yaw_t)
        # print(x,y,z,pitch,roll,yaw)


    # show results
    plt.plot(time,x,time,y,time,z)
    plt.show()
    # rt605.plot_joint()
    # rt605.show_freq_response()
    # rt605.plot_cartesian()
    rt605.freq_response()

if __name__ == "__main__":
    main()