# Form implementation generated from reading ui file 'progress_window.ui'
#
# Created by: PyQt6 UI code generator 6.5.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_progress_window(object):
    def setupUi(self, progress_window):
        progress_window.setObjectName("progress_window")
        progress_window.setEnabled(True)
        progress_window.resize(260, 102)
        self.progressBar = QtWidgets.QProgressBar(parent=progress_window)
        self.progressBar.setEnabled(True)
        self.progressBar.setGeometry(QtCore.QRect(10, 70, 241, 23))
        self.progressBar.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.DefaultContextMenu)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.state = QtWidgets.QLabel(parent=progress_window)
        self.state.setGeometry(QtCore.QRect(50, 20, 191, 51))
        font = QtGui.QFont()
        font.setFamily("Alef")
        font.setPointSize(20)
        self.state.setFont(font)
        self.state.setObjectName("state")

        self.retranslateUi(progress_window)
        QtCore.QMetaObject.connectSlotsByName(progress_window)

    def retranslateUi(self, progress_window):
        _translate = QtCore.QCoreApplication.translate
        progress_window.setWindowTitle(_translate("progress_window", "Simulation Progress"))
        self.state.setText(_translate("progress_window", "Running..."))