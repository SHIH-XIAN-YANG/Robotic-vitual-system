from rt605 import RT605
import random
from libs.type_define import *
import pymysql
import json
import numpy as np

arm = RT605()

# 
arm.initialize_model()

#read path into program
path_file_dir = './data/Path/'
path_name = 'XY_circle_path.txt'

arm.load_HRSS_trajectory(path_file_dir+path_name) 


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



upper_limit = [100,100,100,100,100,100]
lower_limit = [10,5,5,5,5,5]

connection = pymysql.connect(
    host= "127.0.0.1",  # Localhost IP address
    port= 3305,          # Default MySQL port
    user= "root",        # MySQL root user (caution: use secure credentials)
    password= "Sam512011", # Replace with your actual password
)

cursor = connection.cursor()
cursor.execute("CREATE DATABASE IF NOT EXISTS bw_mismatch_db;")
cursor.execute("USE bw_mismatch_db;")

table_name = "bw_mismatch_data"
sql = f"""CREATE TABLE IF NOT EXISTS {table_name} (
    id INT AUTO_INCREMENT PRIMARY KEY,
    Gain JSON, -- Kp gain of each joints
    Bandwidth JSON,
    contour_err JSON,
    max_bandwidth INT,
    tracking_err_x JSON,
    tracking_err_y JSON,
    tracking_err_z JSON,
    contour_err_img_path VARCHAR(255)
    
)"""
cursor.execute(sql)
connection.commit()

sql = "SELECT MAX(id)+1 AS highest_id FROM bw_mismatch_data;"
cursor.execute(sql)
current_id = int(cursor.fetchone()[0])



for i in range(150):
    
    kp_gain = [random.uniform(lower, upper) for lower, upper in zip(lower_limit, upper_limit)]
    print(f'{i} : gain {kp_gain}')
    for idx, joint in enumerate(arm.joints):
        joint.setPID(ServoGain.Position.value.kp, kp_gain[idx])

        # joint.setPID(ServoGain.Position.value.kp, upper_limit[idx])
    arm.start()
    arm.freq_response(show=False)
    # fig = arm.plot_polar()


    # arm.save_log('./data/')


    try:
        gain_json = json.dumps([arm.joints[i].pos_amp.kp for i in range(6)])
        bandwidth_json = json.dumps(arm.bandwidth)
        max_bandwidth = np.argmax(arm.bandwidth)+1
        contour_error_json = json.dumps(arm.contour_err)
        tracking_err_x_json = json.dumps(arm.tracking_err_x)
        tracking_err_y_json = json.dumps(arm.tracking_err_y)
        tracking_err_z_json = json.dumps(arm.tracking_err_z)

        
        c_err_img = arm.plot_polar(show=False)
        fig_path = f"C:\\Users\\Samuel\\Desktop\\mismatch_dataset\\{current_id+i}.png"
        c_err_img.savefig(fig_path)

        sql = """INSERT INTO bw_mismatch_data 
                (Gain, Bandwidth, max_bandwidth, contour_err, 
                tracking_err_x, tracking_err_y, tracking_err_z, contour_err_img_path) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"""
        cursor.execute(sql, (gain_json, bandwidth_json, max_bandwidth, contour_error_json, 
                                tracking_err_x_json, tracking_err_y_json, tracking_err_z_json
                                ,fig_path))
        connection.commit()
        
    except Exception as ex:
        print(ex)