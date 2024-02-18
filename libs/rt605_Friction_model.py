from typing import Any
from libs.RobotManipulators import RT605_710
import libs.ServoDriver as dev
import numpy as np
from threading import Thread
from libs.ServoMotor import ServoMotor
from libs.type_define import *
from libs import ControlSystem as cs
from libs import ServoDriver
from libs.ForwardKinematic import FowardKinematic
import json

class RT605_Friction_Model():
    def __init__(self,en=False) -> None:
        self.enabled = en
    
    def enable_friction(self, en:bool=True):
        self.enabled = en

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return 0



