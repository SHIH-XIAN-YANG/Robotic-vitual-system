from rt605 import RT605
import random
from libs.type_define import *
import pymysql
import json
import numpy as np
import sqlite3
from tqdm import tqdm
arm = RT605()

# 
arm.initialize_model()

#read path into program
path_file_dir = './data/Path/'
path_name = 'XY_circle_path.txt'

arm.load_HRSS_trajectory(path_file_dir+path_name) 
arm.forward_kinematic.setOffset([0,0,120]) # Tool path

### Optioinal functionality

# arm.compute_GTorque.mode = False #Ture off gravity
arm.compute_GTorque.enable_Gtorq(True) #Ture on/off gravity
arm.compute_friction.enable_friction(False) #Turn on/off friction



# arm.setPID(0, gain="kpp",value=80)
# gain:
# kpp
# kvp
# kpi
# kvi

# arm.setMotorModel(0, component="Jm", value=0.05)

# xr = readPATH("D:/*.txt")
# x = []
# for i in range(len(xr)):
#     x.append(arm(xr[i]))

# # Start simulation
# arm.start()  #TODO:只吃單點

# # plot the frequency response of six joints
# arm.freq_response(show=False)

# plot cartesian trajectory in cartesian/joints space respectively
# arm.plot_cartesian()
# arm.plot_joint()
# arm.plot_polar()
# arm.plot_torq()

# arm.save_log('./data/')


#define range of PID controllers
upper_limit = [300,300,300,300,300,300]
lower_limit = [1,1,1,1,1,1]

connection = pymysql.connect(
    host= "127.0.0.1",  # Localhost IP address
    port= 3306,          # Default MySQL port
    user= "root",        # MySQL root user (caution: use secure credentials)
    password= "Sam512011", # Replace with your actual password
)
#Connect to SQLite database (will create it if not exists)
# connection = sqlite3.connect('mismatch_db.db')
cursor = connection.cursor()
cursor.execute("CREATE DATABASE IF NOT EXISTS bw_mismatch_db;")
cursor.execute("USE bw_mismatch_db;")

table_name = "bw_mismatch_data_new"
sql = f"""CREATE TABLE IF NOT EXISTS {table_name} (
    id INT AUTO_INCREMENT PRIMARY KEY,
    gain JSON, -- Kp gain of each joints
    bandwidth JSON,
    min_bandwidth INT,
    contour_err JSON,
    phase_delay JSON,
    phase JSON,
    ori_contour_err JSON,
    tracking_err_x JSON,
    tracking_err_y JSON,
    tracking_err_z JSON,
    tracking_err_pitch JSON,
    tracking_err_roll JSON,
    tracking_err_yaw JSON,
    contour_err_img_path VARCHAR(100),
    phase_delay_img_path VARCHAR(100),
    ori_contour_err_img_path VARCHAR(100)
)"""
cursor.execute(sql)
connection.commit()

sql = "SELECT MAX(id)+1 AS highest_id FROM bw_mismatch_data;"
cursor.execute(sql)
id = cursor.fetchone()[0]
if id is None:
    current_id = 1
else:
    current_id = int(id)



for i in tqdm(range(4000)):
    arm.initialize_model()
    arm.load_HRSS_trajectory(path_file_dir+path_name) 
    
    kp_gain = [random.uniform(lower, upper) for lower, upper in zip(lower_limit, upper_limit)]
    # print(f'{i} : gain {kp_gain}')

    for idx, joint in enumerate(arm.joints):
        joint.setPID(ServoGain.Position.value.kp, kp_gain[idx])

        # joint.setPID(ServoGain.Position.value.kp, upper_limit[idx])
    arm.start()

    arm.freq_response(show=False)
    c_err_fig,ori_c_err_fig, phase_delay_fig = arm.plot_polar(show=False)
    # fig = arm.plot_polar()


    # arm.save_log('./data/')
    try:
        gain_json = json.dumps([arm.joints[j].pos_amp.kp for j in range(6)])
        bandwidth_json = json.dumps(arm.bandwidth)
        min_bandwidth = np.argmin(arm.bandwidth)

        contour_error_json = json.dumps(arm.contour_err)
        phase_delay_json = json.dumps(arm.phase_delay)
        ori_contour_error_json = json.dumps(arm.ori_contour_err)
        phase_json = json.dumps(arm.phase)

        tracking_err_x_json = json.dumps(arm.tracking_err_x)
        tracking_err_y_json = json.dumps(arm.tracking_err_y)
        tracking_err_z_json = json.dumps(arm.tracking_err_z)
        tracking_err_pitch_json = json.dumps(arm.tracking_err_pitch)
        tracking_err_roll_json = json.dumps(arm.tracking_err_roll)
        tracking_err_yaw_json = json.dumps(arm.tracking_err_yaw)
        
        c_err_fig_path = f"C:\\Users\\Samuel\\Desktop\\mismatch_dataset\\contour_error\\{current_id+i}.png"
        c_err_fig.savefig(c_err_fig_path)
        ori_c_err_fig_path = f"C:\\Users\\Samuel\\Desktop\\mismatch_dataset\\orientation_contour_error\\{current_id+i}.png"
        ori_c_err_fig.savefig(ori_c_err_fig_path)
        phase_delay_fig_path = f"C:\\Users\\Samuel\\Desktop\\mismatch_dataset\\phase_delay\\{current_id+i}.png"
        phase_delay_fig.savefig(phase_delay_fig_path)


        sql = """INSERT INTO bw_mismatch_data 
                (gain, bandwidth, min_bandwidth, 
                contour_err, ori_contour_err, phase_delay, phase,
                tracking_err_x, tracking_err_y, tracking_err_z, tracking_err_pitch, tracking_err_roll, tracking_err_yaw,
                  contour_err_img_path, ori_contour_err_img_path, phase_delay_img_path) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
        cursor.execute(sql, (gain_json, bandwidth_json, min_bandwidth, contour_error_json, ori_contour_error_json, phase_delay_json, phase_json,
                                tracking_err_x_json, tracking_err_y_json, tracking_err_z_json, tracking_err_pitch_json, tracking_err_roll_json,tracking_err_yaw_json,
                                c_err_fig_path, ori_c_err_fig_path, phase_delay_fig_path))
        connection.commit()
        
    except Exception as ex:
        print(ex)

cursor.close()
connection.close()