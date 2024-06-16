import pymysql
import json
from PIL import Image
import numpy as np
from rt605 import RT605
from libs.type_define import *
from matplotlib import pyplot as plt
class Mismatch_DataBase():
    __host_name:str = "localhost"
    __port:int = 3306
    __user:str = "root"
    __password:str = "Sam512011"
    __db_name:str = "bw_mismatch_db"

    def __init__(self,):
        self.connection = pymysql.connect(
                host= "127.0.0.1",  # Localhost IP address
                port= 3306,          # Default MySQL port
                user= "root",        # MySQL root user (caution: use secure credentials)
                password= "Sam512011", # Replace with your actual password
            )
        self.cursor = self.connection.cursor()

    def connect_dataBase(self):
        """
        Creates a connection to the MySQL database.

        Args:
            host (str, optional): The hostname or IP address of the MySQL server. Defaults to "localhost".
            port (int, optional): The port number of the MySQL server. Defaults to 3306.
            user (str, optional): The username to connect to the MySQL server. Defaults to "root".
            password (str, optional): The password to connect to the MySQL server. **Never store actual passwords in plain text!**
            db_name (str, optional): The name of the database to connect to. Defaults to "bw_mismatch_db".
        """
        try:

            self.cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.__db_name};")
            self.cursor.execute("USE bw_mismatch_db;")
            
            table_name = "bw_mismatch_data"
            sql = f"""CREATE TABLE IF NOT EXISTS {table_name} (
                id INT AUTO_INCREMENT PRIMARY KEY,
                Gain JSON, -- Kp gain of each joints
                Bandwidth JSON,
                contour_err JSON,
                min_bandwidth INT,
                tracking_err_x JSON,
                tracking_err_y JSON,
                tracking_err_z JSON,
                contour_err_img_path VARCHAR(255)
            )"""
            self.cursor.execute(sql)

            self.connection.commit()

        except Exception as ex:
            print(ex)

    def fetch_data(self, filed:str):
        """
        Fetches all data from the `bw_mismatch_data` table.

        Returns:
            list: A list of rows, where each row is a tuple containing the data from the database.
        """

        try:
            self.cursor.execute(f"SELECT {filed} FROM bw_mismatch_data;")
            rows = self.cursor.fetchall()
            self.connection.commit()
            print("Fetched data successfully.")
            return rows
        except pymysql.Error as e:
            print("Error fetching data:", e)
            return None
        
    def write_data(self,table_name:str, field_name:str, value, id:int, type:str):
        try:
            sql = f"UPDATE {table_name} SET {field_name}=%s WHERE id={id};"
            self.cursor.execute(sql,(value,))
            
            self.connection.commit()

        except pymysql.Error as e:
            print("Error writing data:", e)
            return None
    
  
        

    def close_connection(self):
        """
        Closes the connection to the database.
        """

        if self.connection:
            try:
                self.connection.close()
                
                print("Connection closed.")
            except pymysql.Error as e:
                print("Error closing connection:", e)

# my_db = Mismatch_DataBase()

# my_db.connect_dataBase()

# Fetch data from the table
# gain_data = my_db.fetch_data('Gain')
# bw_data = my_db.fetch_data('Bandwidth')
# min_bw_data = my_db.fetch_data('min_bandwidth')

rt605 = RT605()
rt605.initialize_model() # load servo inforamtion

#load path
path_file_dir = './data/Path/'
path_name = "XY_circle_path.txt"

# load_HRSS_trajectory --> Return joint trajectory in column vector (ndarray)
rt605.load_HRSS_trajectory(path_file_dir+path_name) 
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

cursor.execute("CREATE DATABASE IF NOT EXISTS bw_mismatch_db;")
cursor.execute("USE bw_mismatch_db;")

table_name = "bw_mismatch_data_new"
sql = f"""CREATE TABLE IF NOT EXISTS {table_name} (
    id INT AUTO_INCREMENT PRIMARY KEY,
    gain JSON, -- Kp gain of each joints
    bandwidth JSON,
    min_bandwidth INT,
    contour_err JSON,
    circular_err JSON,
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
    ori_contour_err_img_path VARCHAR(100),
    height_contour_err_img_path VARCHAR(100)
)"""
cursor.execute(sql)
connection.commit()

for id in range(0,4000):
    
    sql = f"SELECT Gain FROM bw_mismatch_data_new WHERE id={id+1};"
    cursor.execute(sql)
    
    connection.commit()
    rows = cursor.fetchall()[0]
    
    gain = json.loads(rows[0])
    print(f'{id+1} : {gain}')

    rt605.initialize_model() # load servo inforamtion
    # load_HRSS_trajectory --> Return joint trajectory in column vector (ndarray)
    rt605.load_HRSS_trajectory(path_file_dir+path_name) 
    for i,kp in enumerate(gain):
        rt605.joints[i].setPID(ServoGain.Position.value.kp, kp)
    
    rt605.start()

    # contour_error_json = json.dumps(rt605.contour_err)
    # phase_delay_json = json.dumps(rt605.phase_delay)
    circular_err_json = json.dumps(rt605.circular_err)
    # ori_contour_error_json = json.dumps(rt605.ori_contour_err)
    # phase_json = json.dumps(rt605.phase)

    # tracking_err_x_json = json.dumps(rt605.tracking_err_x)
    # tracking_err_y_json = json.dumps(rt605.tracking_err_y)
    # tracking_err_z_json = json.dumps(rt605.tracking_err_z)
    # tracking_err_pitch_json = json.dumps(rt605.tracking_err_pitch)
    # tracking_err_roll_json = json.dumps(rt605.tracking_err_roll)
    # tracking_err_yaw_json = json.dumps(rt605.tracking_err_yaw)

    # c_err_fig, ori_c_err_fig,phase_delay_fig = rt605.plot_polar(show=False)
    # c_err_fig_path = f"C:\\Users\\Samuel\\Desktop\\mismatch_dataset\\contour_error\\{id+1}.png"
    # c_err_fig.savefig(c_err_fig_path)
    # ori_c_err_fig_path = f"C:\\Users\\Samuel\\Desktop\\mismatch_dataset\\orientation_contour_error\\{id+1}.png"
    # ori_c_err_fig.savefig(ori_c_err_fig_path)
    # phase_delay_fig_path = f"C:\\Users\\Samuel\\Desktop\\mismatch_dataset\\phase_delay\\{id+1}.png"
    # phase_delay_fig.savefig(phase_delay_fig_path)

    circular_polar_fig, _ = rt605.plot_circular_polar_plot(show=False)
    circular_polar_fig_path = f"C:\\Users\\Samuel\\Desktop\\mismatch_dataset\\circular_error\\{id+1}.png"
    circular_polar_fig.savefig(circular_polar_fig_path)
    # z_err_fig_path = f"C:\\Users\\Samuel\\Desktop\\mismatch_dataset\\z_error\\{id+1}.png"
    # z_err_fig.savefig(z_err_fig_path)

    sql = f"""UPDATE bw_mismatch_data_new SET circular_err=%s WHERE id={id+1}"""
    cursor.execute(sql, (circular_err_json))
    connection.commit()

    # sql = f"""UPDATE bw_mismatch_data_new SET
    #             contour_err=%s, circular_err=%s, ori_contour_err=%s, phase_delay=%s, phase=%s,
    #             tracking_err_x=%s, tracking_err_y=%s, tracking_err_z=%s, tracking_err_pitch=%s, tracking_err_roll=%s, tracking_err_yaw=%s,
    #             phase_delay_img_path=%s,  height_contour_err_img_path=%s 
    #             WHERE id={id+1}"""
    # cursor.execute(sql, (contour_error_json, circular_err_json, ori_contour_error_json, phase_delay_json, phase_json,
    #                         tracking_err_x_json, tracking_err_y_json, tracking_err_z_json, tracking_err_pitch_json, tracking_err_roll_json,tracking_err_yaw_json,
    #                         phase_delay_fig_path, z_err_fig_path))
    # connection.commit()

    plt.close(circular_polar_fig)
    # plt.close(z_err_fig)

#     for idx in range(6):
#         q_pos_err = rt605.q_pos_err[:,idx]
# #         # print(q_pos_err)
#         sql = f"UPDATE bw_mismatch_joints_data SET tracking_err_j{idx+1}=%s WHERE id={id+1};"
#         cursor.execute(sql,(json.dumps(q_pos_err.tolist(),)))
        
#         connection.commit()
# id = 728
# sql = f"SELECT contour_err FROM bw_mismatch_data WHERE id={id};"
# cursor.execute(sql)

# connection.commit()
# rows = cursor.fetchall()[0]
# # print(rows)

# c_err = json.loads(rows[0])

# sql = f"SELECT phase FROM bw_mismatch_data WHERE id={id};"
# cursor.execute(sql)

# connection.commit()
# rows = cursor.fetchall()[0]
# # print(rows)

# phase = json.loads(rows[0])

# gain = json.loads(rows[0])
# print(f'{id+1} : {gain}')
# for i,kp in enumerate(gain):
#     rt605.joints[i].setPID(ServoGain.Position.value.kp, kp)

# rt605.start()

# rt605.plot_joint()
# rt605.plot_cartesian()
# rt605.plot_polar()



cursor.close()
connection.close()
# for idx in range(6):
#     q_pos_err = rt605.q_pos_err[:,idx]
# #         # print(q_pos_err)
#     print((q_pos_err.shape))
#     # sql = f"UPDATE bw_mismatch_joints_data SET tracking_err_j{idx+1}=%s WHERE id={id+1};"
#     # cursor.execute(sql,(json.dumps(q_pos_err.tolist(),)))

#     # connection.commit()
# rt605.plot_error()




# for id, row in enumerate(gain_data):



# for id, row in enumerate(gain_data):
#     gain = json.loads(row[0])
#     for i,kp in enumerate(gain):
#         rt605.joints[i].setPID(ServoGain.Position.value.kp, kp)
#     print(id)

#     rt605.freq_response_v2(show=False)
#     # print(rt605.bandwidth)

#     # my_db.write_data('Bandwidth',json.dumps(rt605.bandwidth),id+1, type='JSON')

#     # my_db.write_data('min_bandwidth',np.argmin(rt605.bandwidth),id+1, type='JSON')

#     rt605.start()



#     for idx in range(rt605.q_pos_err.shape[1]):
#         q_pos_err = rt605.q_pos_err[:,idx]
#         # print(q_pos_err)
#         my_db.write_data('bw_mismatch_joints_data',f'tracking_err_j{idx+1}' ,json.dumps(q_pos_err.tolist()), id+1, type='JSON')
    
#     sql = """INSERT INTO bw_mismatch_joints_data
#             (tracking_err_j1, tracking_err_j2, tracking_err_j3, tracking_err_j4, tracking_err_j5, tracking_err_j6, Gain, Bandwidth, min_bandwidth, max_bandwidth) 
#             VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"""

#     cursor.execute(sql, ()) 
    # c_err,ori_c_err = rt605.plot_polar_test()
    # fig_path = f"C:\\Users\\Samuel\\Desktop\\mismatch_dataset\\contour_error\\{id+1}.png"
    
    # my_db.write_data('contour_err_img_path',fig_path,id+1, type='VARCHAR')
    # c_err.savefig(fig_path)

    # fig_path = f"C:\\Users\\Samuel\\Desktop\\mismatch_dataset\\orientation_contour_error\\{id+1}.png"
    # ori_c_err.savefig(fig_path)
    # my_db.write_data('ori_contour_err',json.dumps(rt605.ori_contour_err),id+1, type='JSON')
    # my_db.write_data('tracking_err_pitch',json.dumps(rt605.tracking_err_pitch), id+1, type='JSON')
    # my_db.write_data('tracking_err_roll',json.dumps(rt605.tracking_err_roll), id+1, type='JSON')
    # my_db.write_data('tracking_err_yaw',json.dumps(rt605.tracking_err_yaw), id+1, type='JSON')

# print(f'gain {gain}')
cursor.close()
connection.close()