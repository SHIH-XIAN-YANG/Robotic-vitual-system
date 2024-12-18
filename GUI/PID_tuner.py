# Form implementation generated from reading ui file 'PID_tuner.ui'
#
# Created by: PyQt6 UI code generator 6.5.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.

import sys

from PyQt6 import QtCore, QtGui, QtWidgets
from PID_tuner_GUI import Ui_RT605_simulation  # Replace "PID_tuner" with the actual filename containing the UI code
from progress_window import Ui_progress_window
from ..rt605 import RT605
from PyQt6.QtWidgets import QFileDialog, QProgressBar
from multiprocessing import Process

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_RT605_simulation()
        self.ui.setupUi(self)
        self.progress_ui = Ui_progress_window()
        self.progress_ui.setupUi(self)
        # self.progress_ui.hide()


        # Connect button signals to custom functions (optional)
        self.ui.load_servo_para_button.clicked.connect(self.load_servo_parameters)
        self.ui.load_HRSS_button.clicked.connect(self.load_hrss_intp)
        self.ui.links_chooser.currentIndexChanged.connect(self.link_chooser_state_change)
        self.ui.run_button.clicked.connect(self.start_simulation)
        self.ui.export_param_button.clicked.connect(self.export_servo_param)

        # parameter setting
        self.ui.deg2pulse_button.clicked.connect(self.set_deg2pulse)
        self.ui.kp_

        
        # parameter already import flag
        self.param_ready_flag = False # state whether servo parameter been loaded or not
        self.hrss_ready_flag = False # state whether hrss interpolaiton been loaded or not

        # links state(changed by combobox)
        self.link_state = 1 # default: link1

        self.rt605 = RT605()
        

        self.show()

    def load_servo_parameters(self):
        # TODO add json to read
        self.rt605.initialize_model()
        self.param_ready_flag = True
        self.ui.servo_param_loaded_radio_button.setStyleSheet('QRadioButton { color: green; }')
        if(self.hrss_ready_flag and self.param_ready_flag):
            self.ui.run_button.setEnabled(True)
        
        # set default link state
        self.ui.link_state_lable.setText('Link 1')
        self.link_state = 1
        
        

    def load_hrss_intp(self):
        # load HRSS interpolation path
        filename, _ = QFileDialog.getOpenFileName(self, "Select a file", "./data/Path/", "All Files (*);;Text Files (*.txt)")        
        data = self.rt605.load_HRSS_trajectory(filename)
        if data is not None:
            print('path loaded')
            self.hrss_ready_flag = True
            self.ui.hrss_loaded_radio_button.setStyleSheet('QRadioButton { color: green; }')
        else:
            print('load path error')
        
        if(self.hrss_ready_flag and self.param_ready_flag):
            self.ui.run_button.setEnabled(True)

    def link_chooser_state_change(self):
        # choose which link parameter to be tuned
        self.link_state = self.ui.links_chooser.currentIndex()+1
        self.ui.link_state_lable.setText(self.ui.links_chooser.currentText())

    def progress_update(self):
        
        self.statusBar().showMessage("Ready", 0)
        self.statusBar().addPermanentWidget(self.ui.progress_bar)
        self.ui.progress_bar.setFixedSize(self.geometry().width()-120, 16)
        self.ui.progress_bar.show()
        self.statusBar().showMessage('RT605 running...',0)

        while self.rt605.progress<100:
            self.ui.progress_bar.setValue(int(self.rt605.progress))

        self.statusBar().showMessage('Completed', 0)
        self.ui.progress_bar.hide()
        
        

    def start_simulation(self):

        process = Process(target=self.progress_update)
        process.start()
        # start rt605 simulation
        self.rt605.start()

        process.join()


    def export_servo_param(self):
        # export servo parameter to json
        pass

    def set_deg2pulse(self):
        pass

    def set_kp(self):
        pass

    def set_vel_limiter(self):
        pass

    def sest_vel_cmd_filter(self):
        pass

    def sest_kv(self):
        pass

    def sest_torq_limiter(self):
        pass

    def sest_torq_filter(self):
        pass

    def sest_motor(self):
        pass

    def sest_reducer(self):
        pass



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())