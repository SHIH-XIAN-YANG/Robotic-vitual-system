import pymysql
import json
from PIL import Image
import numpy as np
from rt605 import RT605
from libs.type_define import *
from matplotlib import pyplot as plt


rt605 = RT605()
rt605.initialize_model() # load servo inforamtion

#load path
path_file_dir = './data/Path/'
path_name = "XY_circle_path.txt"

# load_HRSS_trajectory --> Return joint trajectory in column vector (ndarray)
q_c = rt605.load_HRSS_trajectory(path_file_dir+path_name) 
rt605.forward_kinematic.setOffset([0,0,120])

# example enable gravity effect
rt605.compute_GTorque.enable_Gtorq(en=True)
rt605.compute_friction.enable_friction(en=True)
connection = pymysql.connect(
    host= "127.0.0.1",  # Localhost IP address
    port= 3306,          # Default MySQL port
    user= "root",        # MySQL root user (caution: use secure credentials)
    password= "Sam512011", # Replace with your actual password
    database='bw_mismatch_db'
)
cursor = connection.cursor()

id= 10

sql = f"SELECT gain FROM bw_mismatch_data WHERE id={id};"
cursor.execute(sql)

connection.commit()
rows = cursor.fetchall()[0]

gain = json.loads(rows[0])

# for i in range(5):
# print(i)
for i,kp in enumerate(gain):
    rt605.joints[i].setPID(ServoGain.Position.value.kp, kp)

rt605.start()

# rt605.plot_polar(True)
rt605.plot_cartesian()
plt.plot(rt605.phase)
plt.grid(True)
plt.show()
plt.plot(rt605.contour_err)
plt.grid(True)
plt.show()


# print(phase)
# plt.plot(rt605.phase)
# plt.grid()
# plt.show()