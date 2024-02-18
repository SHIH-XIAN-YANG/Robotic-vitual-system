# Form implementation generated from reading ui file 'PID_tuner.ui'
#
# Created by: PyQt6 UI code generator 6.5.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.

import sys
sys.path.append('../')
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')

from rt605 import RT605
from libs.ServoMotor import ServoMotor
from libs.type_define import *

# print(sys.path)
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtGui import *
from PyQt6.QtCore import QThread, pyqtSignal,Qt, QTimer
from PID_tuner_GUI import Ui_RT605_simulation  # Replace "PID_tuner" with the actual filename containing the UI code
from progress_window import Ui_progress_window
from PyQt6.QtWidgets import QFileDialog, QProgressBar, QMessageBox
from multiprocessing import Process
import time
import os
import threading

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

"""
--------------------------------import pop-out UI----------------------------
"""
from deg2pulse_GUI import Ui_deg2pulse_dialog
from kp_gain_GUI import Ui_kp_setting_dialog
from kv_gain_GUI import Ui_kv_setting_dialog
from motor_GUI import Ui_motor_setting_dialog
from reducer_GUI import Ui_reducer_dialog
from torq_cmd_filter_GUI import Ui_torq_cmd_filter_dialog
from torq_cmd_limiter_GUI import Ui_torq_cmd_limiter_dialog
from vel_cmd_filter_GUI import Ui_vel_cmd_filter_dialog
from vel_cmd_limiter_GUI import Ui_vel_cmd_limiter_dialog




class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_RT605_simulation()
        self.ui.setupUi(self)
        self.progress_window = QtWidgets.QMainWindow()
        self.progress_ui = Ui_progress_window()
        self.progress_ui.setupUi(self.progress_window)
        self.progress_updated_signal = pyqtSignal(int)
        # self.progress_window.show()
        
        #  parameter setting Dialog
        self.kp_setting_dialog = QtWidgets.QDialog()
        self.kp_setting_ui = Ui_kp_setting_dialog()
        self.kp_setting_ui.setupUi(self.kp_setting_dialog)
        self.kp_setting_ui.buttonBox.accepted.connect(self.set_kp)
        # self.kp_setting_ui.buttonBox.rejected.connect(self.kp_setting_ui)

        self.vel_limiter_setting_dialog = QtWidgets.QDialog()
        self.vel_limiter_setting_ui = Ui_vel_cmd_limiter_dialog()
        self.vel_limiter_setting_ui.setupUi(self.vel_limiter_setting_dialog)
        self.vel_limiter_setting_ui.buttonBox.accepted.connect(self.set_vel_limiter)


        self.vel_cmd_filter_setting_dialog = QtWidgets.QDialog()
        self.vel_cmd_filter_setting_ui = Ui_vel_cmd_filter_dialog()
        self.vel_cmd_filter_setting_ui.setupUi(self.vel_cmd_filter_setting_dialog)
        self.vel_cmd_filter_setting_ui.buttonBox.accepted.connect(self.set_vel_cmd_filter)

        self.kv_setting_dialog = QtWidgets.QDialog()
        self.kv_setting_ui = Ui_kv_setting_dialog()
        self.kv_setting_ui.setupUi(self.kv_setting_dialog)
        self.kv_setting_ui.buttonBox.accepted.connect(self.set_kv)
        
        self.torq_cmd_limiter_setting_dialog = QtWidgets.QDialog()
        self.torq_cmd_limiter_setting_ui = Ui_torq_cmd_limiter_dialog()
        self.torq_cmd_limiter_setting_ui.setupUi(self.torq_cmd_limiter_setting_dialog)
        self.torq_cmd_limiter_setting_ui.buttonBox.accepted.connect(self.set_torq_cmd_limiter)

        self.torq_cmd_filter_setting_dialog = QtWidgets.QDialog()
        self.torq_cmd_filter_setting_ui = Ui_torq_cmd_filter_dialog()
        self.torq_cmd_filter_setting_ui.setupUi(self.torq_cmd_filter_setting_dialog)
        self.torq_cmd_filter_setting_ui.buttonBox.accepted.connect(self.set_torq_cmd_filter)

        self.motor_setting_dialog = QtWidgets.QDialog()
        self.motor_setting_ui = Ui_motor_setting_dialog()
        self.motor_setting_ui.setupUi(self.motor_setting_dialog)
        self.motor_setting_ui.buttonBox.accepted.connect(self.set_motor)

        
        # Connect button signals to custom functions (optional)
        self.ui.load_servo_param_button.clicked.connect(self.load_servo_parameters)
        self.ui.load_HRSS_button.clicked.connect(self.load_hrss_intp)
        self.ui.links_chooser.currentIndexChanged.connect(self.link_chooser_state_change)
        self.ui.run_button.clicked.connect(self.start_simulation)
        self.ui.export_param_button.clicked.connect(self.export_servo_param)
        self.ui.freq_response_button.clicked.connect(self.show_freq_response)

        # parameter setting
        # self.ui.deg2pulse_button.clicked.connect(self.set_deg2pulse)
        self.ui.kp_button.clicked.connect(self.set_kp_button_handler)
        self.ui.vel_limiter_button.clicked.connect(self.set_vel_limiter_button_handler)
        self.ui.vel_command_filter_button.clicked.connect(self.set_vel_cmd_filter_button_handler)
        self.ui.kv_button.clicked.connect(self.set_kv_button_handler)
        self.ui.torq_limiter_button.clicked.connect(self.set_torq_cmd_limiter_button_handler)
        self.ui.torq_command_filter_button.clicked.connect(self.set_torq_cmd_filter_button_handler) 
        self.ui.motor_button.clicked.connect(self.set_motor_button_handler)
        self.ui.kp_button.enterEvent = lambda enterEvent: self.ui.kp_button.setIcon(QIcon('./icon/Kp_activated.png'))
        self.ui.kp_button.leaveEvent = lambda leaveEvent: self.ui.kp_button.setIcon(QIcon('./icon/Kp.png'))
        self.ui.vel_limiter_button.enterEvent = lambda enterEvent: self.ui.vel_limiter_button.setIcon(QIcon('./icon/limiter_activated.png'))
        self.ui.vel_limiter_button.leaveEvent = lambda leaveEvent: self.ui.vel_limiter_button.setIcon(QIcon('./icon/limiter.png'))
        self.ui.vel_command_filter_button.enterEvent = lambda enterEvent: self.ui.vel_command_filter_button.setIcon(QIcon('./icon/filter_activated.png'))
        self.ui.vel_command_filter_button.leaveEvent = lambda leaveEvent: self.ui.vel_command_filter_button.setIcon(QIcon('./icon/filter.png'))
        self.ui.kv_button.enterEvent = lambda enterEvent: self.ui.kv_button.setIcon(QIcon('./icon/Kv_activated.png'))
        self.ui.kv_button.leaveEvent = lambda leaveEvent: self.ui.kv_button.setIcon(QIcon('./icon/Kv.png'))
        self.ui.torq_limiter_button.enterEvent = lambda enterEvent: self.ui.torq_limiter_button.setIcon(QIcon('./icon/limiter_activated.png'))
        self.ui.torq_limiter_button.leaveEvent = lambda leaveEvent: self.ui.torq_limiter_button.setIcon(QIcon('./icon/limiter.png'))
        self.ui.torq_command_filter_button.enterEvent = lambda enterEvent: self.ui.torq_command_filter_button.setIcon(QIcon('./icon/filter_activated.png'))
        self.ui.torq_command_filter_button.leaveEvent = lambda leaveEvent: self.ui.torq_command_filter_button.setIcon(QIcon('./icon/filter.png'))
        self.ui.motor_button.enterEvent = lambda enterEvent: self.ui.motor_button.setIcon(QIcon('./icon/motor_activated.png'))
        self.ui.motor_button.leaveEvent = lambda leaveEvent: self.ui.motor_button.setIcon(QIcon('./icon/motor.png'))
        # self.ui.reducer_button.clicked.connect(self.set_reducer)

        # add reference to other dialog
        self.dialogs = [self.kp_setting_dialog,
                        self.vel_cmd_filter_setting_dialog,
                        self.vel_limiter_setting_dialog,
                        self.kv_setting_dialog,
                        self.torq_cmd_filter_setting_dialog,
                        self.torq_cmd_limiter_setting_dialog,
                        self.motor_setting_dialog]

        
        # parameter already import flag
        self.param_ready_flag = False # state whether servo parameter been loaded or not
        self.hrss_ready_flag = False # state whether hrss interpolaiton been loaded or not

        # links state(changed by combobox)
        self.link_state = 1 # default: link1

        self.rt605 = RT605()
        
        # Figure canvas
        self.freq_graphicScene = QtWidgets.QGraphicsScene()
        self.joint_graphicScene = QtWidgets.QGraphicsScene()
        self.cartesian_graphicScene = QtWidgets.QGraphicsScene()
        self.polar_graphicScene = QtWidgets.QGraphicsScene()

        self.default_setup()

        self.show()

    def default_setup(self):
        # load default path(XY_Circle_Paht.txt) and servo parameter
        self.rt605.initialize_model('./../data/servos/')
        self.rt605.load_HRSS_trajectory('./../data/Path/XY_circle_path.txt')
        
        self.param_ready_flag = True
        self.hrss_ready_flag = True
        self.link_state = 1
        self.ui.run_button.setEnabled(True)
        self.ui.freq_response_button.setEnabled(True)
        self.ui.links_chooser.setEnabled(True)

         # enable servo tuning button
        self.ui.deg2pulse_button.setEnabled(True)
        self.ui.kp_button.setEnabled(True)
        self.ui.vel_limiter_button.setEnabled(True)
        self.ui.vel_command_filter_button.setEnabled(True)
        self.ui.kv_button.setEnabled(True)
        self.ui.torq_limiter_button.setEnabled(True)
        self.ui.torq_command_filter_button.setEnabled(True)
        self.ui.motor_button.setEnabled(True)
        self.ui.reducer_button.setEnabled(True)

    def load_servo_parameters(self):
        # TODO add json to read
        self.rt605.initialize_model('./../data/servos/')
        self.param_ready_flag = True
        self.ui.servo_param_loaded_checkBox.setChecked(True)
        if(self.hrss_ready_flag and self.param_ready_flag):
            self.ui.run_button.setEnabled(True)
            self.ui.freq_response_button.setEnabled(True)
        self.ui.links_chooser.setEnabled(True)
        
        
        # set default link state
        self.link_state = 1

        # enable servo tuning button
        self.ui.deg2pulse_button.setEnabled(True)
        self.ui.kp_button.setEnabled(True)
        self.ui.vel_limiter_button.setEnabled(True)
        self.ui.vel_command_filter_button.setEnabled(True)
        self.ui.kv_button.setEnabled(True)
        self.ui.torq_limiter_button.setEnabled(True)
        self.ui.torq_command_filter_button.setEnabled(True)
        self.ui.motor_button.setEnabled(True)
        self.ui.reducer_button.setEnabled(True)


        print('servo parameters loaded successfully')
        
    def load_hrss_intp(self):
        # load HRSS interpolation path
        filename, _ = QFileDialog.getOpenFileName(self, "Select a file", "./../data/Path/", "All Files (*);;Text Files (*.txt)")        
        data = self.rt605.load_HRSS_trajectory(filename)
        if data is not None:
            print('path loaded successfully')
            self.hrss_ready_flag = True
            self.ui.hrss_loaded_checkBox.setChecked(True)
            self.ui.path_name.setText(os.path.splitext(os.path.basename(filename))[0])
        else:
            print('load path error')
        
        if(self.hrss_ready_flag and self.param_ready_flag):
            self.ui.run_button.setEnabled(True)
            self.ui.freq_response_button.setEnabled(True)

    def link_chooser_state_change(self):
        # choose which link parameter to be tuned
        self.link_state = self.ui.links_chooser.currentIndex()

    def start_simulation(self):
        print('start simulation...')
        self.ui.run_button.setEnabled(False)
        # simulation_thread = threading.Thread(target=self.rt605.start)
        # simulation_thread.start()
        # simulation_thread.join()
        
        self.rt605.start()

        print('finished')
        
        self.ui.run_button.setEnabled(True)

        # show the cartesian result
        self.show_carteisan3D_plot()
        self.show_joint_plot()
        self.show_polar_plot()



    def export_servo_param(self):
        # export servo parameter to json
        pass

    def set_kp_button_handler(self):
        self.kp_setting_ui.unit_label.setText(str(self.rt605.joints[self.link_state].pos_amp.kp_unit))
        self.kp_setting_ui.current_value_label.setText(str(self.rt605.joints[self.link_state].pos_amp.kp))
        self.kp_setting_ui.lineEdit.setText(str(self.rt605.joints[self.link_state].pos_amp.kp))
        self.kp_setting_dialog.show()
        

    def set_kp(self):
        self.rt605.joints[self.link_state].setPID(ServoGain.Position.value.kp, float(self.kp_setting_ui.lineEdit.text()))
        self.kp_setting_ui.current_value_label.setText(self.kp_setting_ui.lineEdit.text())

    def set_vel_limiter_button_handler(self):
        self.vel_limiter_setting_ui.unit_label.setText(str(self.rt605.joints[self.link_state].vel_cmd_lim.unit))
        self.vel_limiter_setting_ui.current_value_label.setText(str(self.rt605.joints[self.link_state].vel_cmd_lim_val))
        self.vel_limiter_setting_ui.lineEdit.setText(str(self.rt605.joints[self.link_state].vel_cmd_lim_val))
        self.vel_limiter_setting_dialog.show()

    def set_vel_limiter(self):
        value = float(self.vel_limiter_setting_ui.lineEdit.text())
        self.rt605.joints[self.link_state].vel_cmd_lim_val = value
        self.rt605.joints[self.link_state].vel_cmd_lim.setLimitation((-1*value, value))
        self.vel_limiter_setting_ui.current_value_label.setText(str(value))


    def set_vel_cmd_filter_button_handler(self):
        self.vel_cmd_filter_setting_ui.unit_label.setText(str(self.rt605.joints[self.link_state].vel_cmd_filter.unit))
        self.vel_cmd_filter_setting_ui.lineEdit.setText(str(self.rt605.joints[self.link_state].vel_cmd_filter.time_const))
        self.vel_cmd_filter_setting_ui.current_value_label.setText(str(self.rt605.joints[self.link_state].vel_cmd_filter.time_const))
        self.vel_cmd_filter_setting_dialog.show()

    def set_vel_cmd_filter(self):
        value = float(self.vel_cmd_filter_setting_ui.lineEdit.text())
        self.rt605.joints[self.link_state].vel_cmd_filter.setup(value)
        self.vel_cmd_filter_setting_ui.current_value_label.setText(str(value))

    def set_kv_button_handler(self):
        self.kv_setting_ui.kvp_unit_label.setText(self.rt605.joints[self.link_state].vel_amp.kp_unit)
        self.kv_setting_ui.tvi_unit_label.setText(self.rt605.joints[self.link_state].vel_amp.ki_unit)
        self.kv_setting_ui.kvp_lineEdit.setText(str(self.rt605.joints[self.link_state].vel_amp.kp))
        self.kv_setting_ui.tvi_lineEdit.setText(str(self.rt605.joints[self.link_state].vel_amp.ki))
        self.kv_setting_ui.kvp_current_value_label.setText(str(self.rt605.joints[self.link_state].vel_amp.kp))
        self.kv_setting_ui.tvi_current_value_label.setText(str(self.rt605.joints[self.link_state].vel_amp.ki))

        self.kv_setting_dialog.show()

    def set_kv(self):
        kvp_value = float(self.kv_setting_ui.kvp_lineEdit.text())
        tvi_value = float(self.kv_setting_ui.tvi_lineEdit.text())

        self.kv_setting_ui.kvp_current_value_label.setText(str(kvp_value))
        self.kv_setting_ui.tvi_current_value_label.setText(str(tvi_value))

        self.rt605.joints[self.link_state].vel_amp.kp = kvp_value
        self.rt605.joints[self.link_state].vel_amp.ki = tvi_value
        

    def set_torq_cmd_limiter_button_handler(self):
        self.torq_cmd_limiter_setting_ui.unit_label.setText(str(self.rt605.joints[self.link_state].tor_cmd_lim.unit))
        self.torq_cmd_limiter_setting_ui.current_value_label.setText(str(self.rt605.joints[self.link_state].tor_cmd_lim_val))
        self.torq_cmd_limiter_setting_ui.lineEdit.setText(str(self.rt605.joints[self.link_state].tor_cmd_lim_val))


        self.torq_cmd_limiter_setting_dialog.show()

    def set_torq_cmd_limiter(self):
        value = float(self.torq_cmd_limiter_setting_ui.lineEdit.text())
        self.torq_cmd_limiter_setting_ui.current_value_label.setText(str(value))
        self.rt605.joints[self.link_state].tor_cmd_lim_val = value
        self.rt605.joints[self.link_state].tor_cmd_lim.setLimitation((-1*value, value))



    def set_torq_cmd_filter_button_handler(self):
        self.torq_cmd_filter_setting_ui.unit_label.setText(self.rt605.joints[self.link_state].tor_cmd_filter.unit)
        self.torq_cmd_filter_setting_ui.lineEdit.setText(str(self.rt605.joints[self.link_state].tor_cmd_filter.get_fc()))
        self.torq_cmd_filter_setting_ui.current_value_label.setText(str(self.rt605.joints[self.link_state].tor_cmd_filter.get_fc()))

        self.torq_cmd_filter_setting_dialog.show()

    def set_torq_cmd_filter(self):
        value = float(self.torq_cmd_filter_setting_ui.lineEdit.text())
        self.rt605.joints[self.link_state].tor_cmd_filter.set_fc(value)
        self.torq_cmd_filter_setting_ui.current_value_label.setText(str(value))

    def set_motor_button_handler(self):
        self.motor_setting_ui.j_unit_label.setText(self.rt605.joints[self.link_state].motor.Jm_unit)
        self.motor_setting_ui.b_unit_label.setText(self.rt605.joints[self.link_state].motor.fric_vis_unit)

        self.motor_setting_ui.j_lineEdit.setText(str(self.rt605.joints[self.link_state].motor.Jm))
        self.motor_setting_ui.b_lineEdit.setText(str(self.rt605.joints[self.link_state].motor.fric_vis))

        self.motor_setting_ui.j_current_value_label.setText(str(self.rt605.joints[self.link_state].motor.Jm))
        self.motor_setting_ui.b_current_value_label.setText(str(self.rt605.joints[self.link_state].motor.fric_vis))

        self.motor_setting_dialog.show()

    def set_motor(self):
        j_value = float(self.motor_setting_ui.j_lineEdit.text())
        b_value = float(self.motor_setting_ui.b_lineEdit.text())

        self.rt605.joints[self.link_state].motor.Jm = j_value
        self.rt605.joints[self.link_state].motor.fric_vis = b_value

        self.motor_setting_ui.j_current_value_label.setText(str(j_value))
        self.motor_setting_ui.b_current_value_label.setText(str(b_value))

    def set_reducer_button_handler(self):
        pass

    def set_reducer(self):
        pass

    def set_deg2pulse_button_handler(self):
        pass

    def set_deg2pulse(self):
        pass

    

    def show_carteisan3D_plot(self):
        _, cartesian3D_fig = self.rt605.plot_cartesian(show=False)
        canvas = FigureCanvas(cartesian3D_fig)
        self.cartesian_graphicScene.addWidget(canvas)

        self.ui.cartesian3D_graph.setScene(self.cartesian_graphicScene)

    def show_freq_response(self):
        bode_fig = self.rt605.freq_response(show=False)
        canvas = FigureCanvas(bode_fig)
        self.freq_graphicScene.addWidget(canvas)
        self.ui.freq_response.setScene(self.freq_graphicScene)

    def show_joint_plot(self):
        joint_fig = self.rt605.plot_joint(show=False)
        canvas = FigureCanvas(joint_fig)
        self.joint_graphicScene.addWidget(canvas)
        self.ui.joint_graph.setScene(self.joint_graphicScene)

    def show_polar_plot(self):
        polar_fig = self.rt605.plot_polar(show=False)
        canvas = FigureCanvas(polar_fig)
        self.polar_graphicScene.addWidget(canvas)
        self.ui.polar_graph.setScene(self.polar_graphicScene)

    def hover_event(self):
        self.ui.kp_button.leaveEvent()
        self.ui.kp_button.enterEvent()

    def closeEvent(self, event):
        reply = QtWidgets.QMessageBox.question(self, 'Exit', 'Are you sure you want to exit?',
                                               QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                                               QtWidgets.QMessageBox.StandardButton.No)
        
        

        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            for dialog in self.dialogs:
                if dialog and dialog.isVisible():
                    dialog.close()
            event.accept()
        else:
            event.ignore()



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())