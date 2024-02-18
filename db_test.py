import pymysql
import json
from PIL import Image


class Mismatch_DataBase():
    __host_name:str = "localhost"
    __port:int = 3306
    __user:str = "root"
    __password:str = "Sam512011"
    __db_name:str = "bw_mismatch_db"

    def __init__(self,):
        self.connection = pymysql.connect(
                host= "127.0.0.1",  # Localhost IP address
                port= 3305,          # Default MySQL port
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
                max_bandwidth INT,
                tracking_err_x JSON,
                tracking_err_y JSON,
                tracking_err_z JSON,
                contour_err_img_path VARCHAR(255)
            )"""
            self.cursor.execute(sql)

            self.connection.commit()

        except Exception as ex:
            print(ex)

    def fetch_data(self):
        """
        Fetches all data from the `bw_mismatch_data` table.

        Returns:
            list: A list of rows, where each row is a tuple containing the data from the database.
        """

        try:
            self.cursor.execute("SELECT * FROM bw_mismatch_data")
            rows = self.cursor.fetchall()
            self.connection.commit()
            print("Fetched data successfully.")
            return rows
        except pymysql.Error as e:
            print("Error fetching data:", e)
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

my_db =Mismatch_DataBase()

my_db.connect_dataBase()

# Fetch data from the table
data = my_db.fetch_data()

gain = json.loads(data[0][1])
bandwidth =  json.loads(data[0][2])
max_bandwidth = data[0][4]
contour_error_img_path = data[0][8]

print(f'gain {gain}')
print(f'bandwidth {bandwidth}')
print(f'max bandwidth {max_bandwidth}')
contour_error_img= Image.open(contour_error_img_path)
contour_error_img.show()
# Process the fetched data (e.g., print, store in a variable)

# Close the connection
my_db.close_connection()
