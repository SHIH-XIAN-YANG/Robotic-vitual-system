# Form implementation generated from reading ui file './ui/vel_cmd_filter.ui'
#
# Created by: PyQt6 UI code generator 6.5.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_vel_cmd_filter_dialog(object):
    def setupUi(self, vel_cmd_filter_dialog):
        vel_cmd_filter_dialog.setObjectName("vel_cmd_filter_dialog")
        vel_cmd_filter_dialog.resize(521, 243)
        self.buttonBox = QtWidgets.QDialogButtonBox(parent=vel_cmd_filter_dialog)
        self.buttonBox.setGeometry(QtCore.QRect(20, 200, 481, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.StandardButton.Cancel|QtWidgets.QDialogButtonBox.StandardButton.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayoutWidget = QtWidgets.QWidget(parent=vel_cmd_filter_dialog)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 10, 499, 181))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label = QtWidgets.QLabel(parent=self.verticalLayoutWidget)
        self.label.setObjectName("label")
        self.horizontalLayout_2.addWidget(self.label)
        self.label_6 = QtWidgets.QLabel(parent=self.verticalLayoutWidget)
        self.label_6.setMaximumSize(QtCore.QSize(300, 50))
        self.label_6.setText("")
        self.label_6.setPixmap(QtGui.QPixmap("./ui\\../icon/vel_cmd_filter.png"))
        self.label_6.setScaledContents(True)
        self.label_6.setObjectName("label_6")
        self.horizontalLayout_2.addWidget(self.label_6)
        self.label_2 = QtWidgets.QLabel(parent=self.verticalLayoutWidget)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.unit_label = QtWidgets.QLabel(parent=self.verticalLayoutWidget)
        self.unit_label.setObjectName("unit_label")
        self.horizontalLayout_2.addWidget(self.unit_label)
        self.horizontalLayout_2.setStretch(0, 2)
        self.horizontalLayout_2.setStretch(1, 2)
        self.horizontalLayout_2.setStretch(2, 1)
        self.horizontalLayout_2.setStretch(3, 2)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_4 = QtWidgets.QLabel(parent=self.verticalLayoutWidget)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout.addWidget(self.label_4)
        self.lineEdit = QtWidgets.QLineEdit(parent=self.verticalLayoutWidget)
        self.lineEdit.setObjectName("lineEdit")
        self.horizontalLayout.addWidget(self.lineEdit)
        self.current_value_label = QtWidgets.QLabel(parent=self.verticalLayoutWidget)
        self.current_value_label.setObjectName("current_value_label")
        self.horizontalLayout.addWidget(self.current_value_label)
        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 5)
        self.horizontalLayout.setStretch(2, 2)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(1, 5)

        self.retranslateUi(vel_cmd_filter_dialog)
        self.buttonBox.accepted.connect(vel_cmd_filter_dialog.accept) # type: ignore
        self.buttonBox.rejected.connect(vel_cmd_filter_dialog.reject) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(vel_cmd_filter_dialog)

    def retranslateUi(self, vel_cmd_filter_dialog):
        _translate = QtCore.QCoreApplication.translate
        vel_cmd_filter_dialog.setWindowTitle(_translate("vel_cmd_filter_dialog", "Setting"))
        self.label.setText(_translate("vel_cmd_filter_dialog", "velocity command filter:\n"
"Currently disabled"))
        self.label_2.setText(_translate("vel_cmd_filter_dialog", "Unit"))
        self.unit_label.setText(_translate("vel_cmd_filter_dialog", "-"))
        self.label_4.setText(_translate("vel_cmd_filter_dialog", "Time constant: "))
        self.current_value_label.setText(_translate("vel_cmd_filter_dialog", "-"))