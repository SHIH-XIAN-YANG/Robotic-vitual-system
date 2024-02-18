#!/bin/bash

ui_dir="./ui/"

# Compile the UI files to Python
# pyuic6 "${ui_dir}PID_tuner.ui" -o "PID_tuner_GUI.py"
# pyuic6 "${ui_dir}deg2pulse.ui" -o "deg2pulse_GUI.py"
# pyuic6 "${ui_dir}kp_gain.ui" -o "kp_gain_GUI.py"
# pyuic6 "${ui_dir}kv_gain.ui" -o "kv_gain_GUI.py"
# pyuic6 "${ui_dir}motor.ui" -o "motor_GUI.py"
# pyuic6 "${ui_dir}reducer.ui" -o "reducer_GUI.py"
# pyuic6 "${ui_dir}torq_cmd_filter.ui" -o "torq_cmd_filter_GUI.py"
# pyuic6 "${ui_dir}torq_cmd_limiter.ui" -o "torq_cmd_limiter_GUI.py"
# pyuic6 "${ui_dir}vel_cmd_filter.ui" -o "vel_cmd_filter_GUI.py"
# pyuic6 "${ui_dir}vel_cmd_limiter.ui" -o "vel_cmd_limiter_GUI.py"

# Check if the compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful."
else
    echo "Error during compilation."
fi

python rt605_simulator.py