"""
This code is for testing the PSO tune the PID controller of RT605

"""

import sys
import os

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from libs.pso import PSO_Algorithm, Particle
from rt605 import RT605
from libs.type_define import *
import numpy as np



class PSO_tune_rt605():
    def __init__(self):
        self.rt605 = RT605()
 
        self.rt605.initialize_model()
        self.rt605.load_HRSS_trajectory('./data/Path/XY_circle_path.txt')
        self.rt605.forward_kinematic.setOffset([0,0,120])
        self.rt605.compute_GTorque.enable_Gtorq(en=True)
        self.rt605.compute_friction.enable_friction(en=True)
        # for i in range(3):
            # self.rt605.joints[i].setPID(ServoGain.Position.value.kp,5)
        self.rt605.start()
        # self.rt605.plot_cartesian()
        # self.rt605.plot_joint()
        # self.rt605.plot_polar()
        # self.rt605.freq_response_v2()
        self.nvar = 12
        self.lower_bound = [0.1]*6 + [0.01]*6
        self.upper_bound = [100]*6 + [1.5]*6
        self.population = 30
        self.iterations = 200


        self.pso:PSO_Algorithm = None

        

    def pso_init(self):
        self.pso = PSO_Algorithm(self.maximize_bw_vel,None, self.nvar, self.population, self.iterations, self.lower_bound, self.upper_bound)
        self.pso.set_max_iteration(max_iter=self.iterations)
        self.pso.set_population(population=self.population)

        self.pso.set_inertia_damping_mode(True)
        self.pso.set_plot_mode(True)
        self.pso.init_particles()

    def pso_start(self):
        self.pso.update()
        # self.rt605.plot_cartesian(True)
        # self.rt605.plot_joint(True)
        # self.rt605.plot_polar(True)
        # self.rt605.plot_polar(True)
        # self.rt605.freq_response(show=True)

    def objective_func(self, args):
        """
        args --> [Kp1,Kp2,Kp3]

        """
        # self.rt605.initialize_model()
        # self.rt605.load_HRSS_trajectory('./data/Path/XY_circle_path.txt')
        # self.rt605.compute_GTorque.enable_Gtorq(en=True)
        # self.rt605.compute_friction.enable_friction(en=True)
        # print(f'obj function: {args}')
        # set Kp gain for link1~link3
        for i in range(self.nvar):
            self.rt605.joints[i].setPID(ServoGain.Position.value.kp, args[i])

        return self.rt605.pso_tune_gain_update()
        
    def save_gain(self):
        for i, joint in enumerate(self.rt605.joints):
            if i < 6:
                joint.setPID(ServoGain.Velocity.value.kp, self.pso.g_best[i])
            else:
                joint.setPID(ServoGain.Velocity.value.ki, self.pso.g_best[i+6])
        
        for i, joint in enumerate(self.rt605.joints):
            joint.export_servo_gain(f'j{i+1}_optimal_gain.gain')
        
    def constrain(self)->bool:
        for joint in (self.rt605.joints):
            if joint.pos_loop_gm<15 or joint.pos_loop_pm<45 or joint.Mpp>100:
                print('particle over constrain')
                return False
        return True
            

    def bw_diff(self, args):
        bw_diff_total = 0

        for i in range(self.nvar):
            self.rt605.joints[i].setPID(ServoGain.Position.value.kp, args[i])

        self.rt605.freq_response_v2(False)
        for i in range(6):
            for j in range(i+1, 6):
                bw_diff_total += abs(self.rt605.bandwidth[i] - self.rt605.bandwidth[j])

        # self.rt605.start()

        err = 0.98* bw_diff_total + 0.01 * sum(np.abs(self.rt605.contour_err)) + 0.01 * sum(np.abs(self.rt605.ori_contour_err))

        return err

    def maximize_bw(self, args):
        for i in range(self.nvar):
            self.rt605.joints[i].setPID(ServoGain.Position.value.kp, args[i])

        self.rt605.freq_response(False)
        over_constrain = self.constrain()
        
        if over_constrain == False:
            return None
        
        return sum([1/bw for bw in self.rt605.bandwidth])
    
    def maximize_bw_vel(self, args):
        for i, joint in enumerate(self.rt605.joints):
            if i <6:
                joint.setPID(ServoGain.Velocity.value.kp, args[i])
            else:
                joint.setPID(ServoGain.Velocity.value.ki, args[i+6])

        self.rt605.freq_response(mode='vel',show=False)
        bw_diff_total = 0

        for i in range(6):
            for j in range(i+1, 6):
                bw_diff_total += abs(self.rt605.bandwidth[i] - self.rt605.bandwidth[j])

        return 10*sum([1/bw for bw in self.rt605.bandwidth])
        

    def show_results(self):
        for i in range(self.nvar):
            self.rt605.joints[i].setPID(ServoGain.Position.value.kp, self.pso.g_best[i])

        self.rt605.start()
        self.rt605.freq_response()
        self.rt605.plot_cartesian()
        self.rt605.plot_joint()
        self.rt605.plot_polar()
        self.rt605.plot_torq()

if __name__ == '__main__':
    pso_tune_rt605 = PSO_tune_rt605()
    

    pso_tune_rt605.pso_init()

    pso_tune_rt605.pso_start()

    pso_tune_rt605.rt605.freq_response('vel')
    pso_tune_rt605.rt605.freq_response('pos')

    # pso_tune_rt605.pso.plot_particle_history()

    # pso_tune_rt605.show_results()

    pso_tune_rt605.save_gain()

    

