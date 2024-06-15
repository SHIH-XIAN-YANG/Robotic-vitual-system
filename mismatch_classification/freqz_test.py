import numpy as np
import matplotlib.pyplot as plt
import pymysql
import csv
import sys

from rt605 import RT605


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

table_name = "bw_mismatch_data"
sql = f"""CREATE TABLE IF NOT EXISTS {table_name} (
    id INT AUTO_INCREMENT PRIMARY KEY,
    gain JSON, -- Kp gain of each joints
    bandwidth JSON,
    max_bandwidth INT,
    contour_err JSON,
    ori_contour_err JSON,
    tracking_err_x JSON,
    tracking_err_y JSON,
    tracking_err_z JSON,
    tracking_err_pitch JSON,
    tracking_err_roll JSON,
    tracking_err_yaw JSON,
    contour_err_img_path VARCHAR(100),
    ori_contour_err_img_path VARCHAR(100)
)"""
cursor.execute(sql)
connection.commit()

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


upper_limit = [200,150,200,200,200,200]
lower_limit = [50,10,1,1,1,1]