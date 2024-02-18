from libs.pso import PSO_Algorithm, Particle
from rt605 import RT605
from libs.type_define import *

class PSO_tune_rt605():
    def __init__(self):
        self.rt605 = RT605()

        
        self.rt605.initialize_model()
        self.rt605.load_HRSS_trajectory('./data/Path/XY_circle_path.txt')
        self.rt605.compute_GTorque.enable_Gtorq(en=True)
        self.rt605.compute_friction.enable_friction(en=True)
        # for i in range(3):
            # self.rt605.joints[i].setPID(ServoGain.Position.value.kp,5)
        # self.rt605.start()
        # self.rt605.plot_cartesian()
        # self.rt605.plot_joint()

        self.nvar = 3
        self.lower_bound = [1,1,1]
        self.upper_bound = [300,300,300]
        self.population = 10
        self.iterations = 100


        self.pso:PSO_Algorithm = None

        

    def pso_init(self):
        self.pso = PSO_Algorithm(self.rosenbrock_function, self.nvar,self.lower_bound, self.upper_bound)
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
        for i in range(3):
            self.rt605.joints[i].setPID(ServoGain.Position.value.kp, args[i])

        return self.rt605.pso_tune_gain_update()
        
    def save_gain(self):
        for i in range(6):
            self.rt605.joints[i].export_servo_gain(f'j{i+1}_optimal_gain.gain')
        
    # def objective_func(self, args):
    #     x = args[0]
    #     y = args[1]
    #     z = args[2]

    #     return (x**2 + y**2 + z**2)
            
    def rosenbrock_function(self, x):
        """
        Rosenbrock function for Particle Swarm Optimization with three variables.
        
        Parameters:
        x (numpy.ndarray): Array containing the values of the three variables.
        
        Returns:
        float: The value of the Rosenbrock function for the given variables.
        """
        a = 1.0
        b = 100.0
        
        x1, x2, x3 = x
        
        term1 = (a - x1)**2
        term2 = b * (x2 - x1**2)**2
        term3 = (a - x2)**2
        term4 = b * (x3 - x2**2)**2
        
        result = term1 + term2 + term3 + term4
        
        return result


if __name__ == '__main__':
    pso_tune_rt605 = PSO_tune_rt605()

    pso_tune_rt605.pso_init()

    pso_tune_rt605.pso_start()

    pso_tune_rt605.pso.plot_particle_history()

    pso_tune_rt605.save_gain()

    

