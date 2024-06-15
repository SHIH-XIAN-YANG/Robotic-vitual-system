from enum import Enum

class Unit(Enum):
    pulse = 1
    rad = 2
    degree = 3
    rev = 4
    rpm = 5
    rps = 6
    
    ms = 14
    ms_0_1 = 15 # 0.1 ms
    sec = 16
    sec_n_1 = 17 # 1/sec.
    Hz = 18
#

class _Position(Enum):
    kp = 1
    ki = 2
    kff = 3
# Velocity Loop:
class _Velocity(Enum):
    kp = 1
    ki = 2
    kff = 3
class ServoGain(Enum):
    # Position Loop:
    Position = _Position
    # Velocity Loop:
    Velocity = _Velocity
#
class MotorModel(Enum):
    Jm = 1
    fric_vis = 2
    fric_c = 3
    fric_dv = 4
#
class FilterType(Enum):
    iir = 1
    fir = 2
    exponential_delay = 3
#
class PidType(Enum):
    P           = 0b0001
    PI          = 0b0011
    PID         = 0b0111
#
class AntiWindupType(Enum):
    def __init__(self):
        pass
#
class MotorType(Enum):
    AC_servo = 1
    DC_servo = 2
#
class MotorDriverType(Enum):
    Yaskawa_SGD7_400W = 1
    Sanyo_RS2_50W = 2
    Sanyo_RS2_100W = 3
    Sanyo_RS2_200W = 4
    Sanyo_RS2_400W = 5
    TM5_DC_servo = 6