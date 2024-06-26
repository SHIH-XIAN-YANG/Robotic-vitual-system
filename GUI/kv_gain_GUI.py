# Form implementation generated from reading ui file './ui/kv_gain.ui'
#
# Created by: PyQt6 UI code generator 6.5.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_kv_setting_dialog(object):
    def setupUi(self, kv_setting_dialog):
        kv_setting_dialog.setObjectName("kv_setting_dialog")
        kv_setting_dialog.resize(400, 280)
        self.buttonBox = QtWidgets.QDialogButtonBox(parent=kv_setting_dialog)
        self.buttonBox.setGeometry(QtCore.QRect(20, 240, 361, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.StandardButton.Cancel|QtWidgets.QDialogButtonBox.StandardButton.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayoutWidget = QtWidgets.QWidget(parent=kv_setting_dialog)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 10, 371, 221))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(parent=self.verticalLayoutWidget)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_2 = QtWidgets.QLabel(parent=self.verticalLayoutWidget)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_4.addWidget(self.label_2)
        self.kvp_unit_label = QtWidgets.QLabel(parent=self.verticalLayoutWidget)
        self.kvp_unit_label.setObjectName("kvp_unit_label")
        self.horizontalLayout_4.addWidget(self.kvp_unit_label)
        self.verticalLayout_2.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_3 = QtWidgets.QLabel(parent=self.verticalLayoutWidget)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_5.addWidget(self.label_3)
        self.tvi_unit_label = QtWidgets.QLabel(parent=self.verticalLayoutWidget)
        self.tvi_unit_label.setObjectName("tvi_unit_label")
        self.horizontalLayout_5.addWidget(self.tvi_unit_label)
        self.verticalLayout_2.addLayout(self.horizontalLayout_5)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        self.horizontalLayout.setStretch(0, 5)
        self.horizontalLayout.setStretch(1, 3)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setContentsMargins(14, -1, -1, -1)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_4 = QtWidgets.QLabel(parent=self.verticalLayoutWidget)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_2.addWidget(self.label_4)
        self.kvp_lineEdit = QtWidgets.QLineEdit(parent=self.verticalLayoutWidget)
        self.kvp_lineEdit.setObjectName("kvp_lineEdit")
        self.horizontalLayout_2.addWidget(self.kvp_lineEdit)
        self.kvp_current_value_label = QtWidgets.QLabel(parent=self.verticalLayoutWidget)
        self.kvp_current_value_label.setObjectName("kvp_current_value_label")
        self.horizontalLayout_2.addWidget(self.kvp_current_value_label)
        self.horizontalLayout_2.setStretch(0, 1)
        self.horizontalLayout_2.setStretch(1, 5)
        self.horizontalLayout_2.setStretch(2, 2)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setContentsMargins(14, -1, -1, -1)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_5 = QtWidgets.QLabel(parent=self.verticalLayoutWidget)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_3.addWidget(self.label_5)
        self.tvi_lineEdit = QtWidgets.QLineEdit(parent=self.verticalLayoutWidget)
        self.tvi_lineEdit.setObjectName("tvi_lineEdit")
        self.horizontalLayout_3.addWidget(self.tvi_lineEdit)
        self.tvi_current_value_label = QtWidgets.QLabel(parent=self.verticalLayoutWidget)
        self.tvi_current_value_label.setObjectName("tvi_current_value_label")
        self.horizontalLayout_3.addWidget(self.tvi_current_value_label)
        self.horizontalLayout_3.setStretch(0, 1)
        self.horizontalLayout_3.setStretch(1, 5)
        self.horizontalLayout_3.setStretch(2, 2)
        self.verticalLayout.addLayout(self.horizontalLayout_3)

        self.retranslateUi(kv_setting_dialog)
        self.buttonBox.accepted.connect(kv_setting_dialog.accept) # type: ignore
        self.buttonBox.rejected.connect(kv_setting_dialog.reject) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(kv_setting_dialog)

    def retranslateUi(self, kv_setting_dialog):
        _translate = QtCore.QCoreApplication.translate
        kv_setting_dialog.setWindowTitle(_translate("kv_setting_dialog", "Setting"))
        self.label.setText(_translate("kv_setting_dialog", "Velocity loop Gain"))
        self.label_2.setText(_translate("kv_setting_dialog", "Kvp unit"))
        self.kvp_unit_label.setText(_translate("kv_setting_dialog", "-"))
        self.label_3.setText(_translate("kv_setting_dialog", "Tvi unit"))
        self.tvi_unit_label.setText(_translate("kv_setting_dialog", "-"))
        self.label_4.setText(_translate("kv_setting_dialog", "Kvp"))
        self.kvp_current_value_label.setText(_translate("kv_setting_dialog", "-"))
        self.label_5.setText(_translate("kv_setting_dialog", "Tvi"))
        self.tvi_current_value_label.setText(_translate("kv_setting_dialog", "-"))
