import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

class Particle:
    def __init__(self,algorithm) -> None:
        self.nvars = algorithm.nvars  # Number of decision variables
        self.position = np.zeros(self.nvars)
        self.velocity = np.zeros(self.nvars)
        self.p_best = np.zeros(self.nvars)
        self.g_best = np.zeros(self.nvars)
        self.p_fitness = None
        self.lr = algorithm.lr
        self.inertia = algorithm.inertia
        self.g_learning_coef = algorithm.g_learning_coef     # Global learning coefficient
        self.p_learning_coef = algorithm.p_learning_coef     # Personal learning coefficietn
        self.obj_func = algorithm.obj_func # Objective function
        self.var_min = algorithm.var_min # Lower bound of Position
        self.var_max = algorithm.var_max # Upper bound of Position
        self.constraints = algorithm.constraints

        self.vel_min = [-0.1  * (a - b) for a, b in zip(self.var_max, self.var_min)] # Velocity limit of particle
        self.vel_max = [-x for x in self.vel_min]
        while self.p_fitness == None:
            # Initailize particle (randomize position)
            for i in range(self.nvars):
                self.position[i] = random.uniform(self.var_min[i],self.var_max[i]) #initialize position
            self.p_fitness = self.fit()
        self.p_best = self.position # Initialize personal best
        
        self.position_history = [] # save the iterative history
    
    # Calculate fitness of particle
    def fit(self): 
        return self.obj_func(self.position) 

    # Update personal best
    def update_p_best(self)->bool:
        fit = self.fit()
        if fit == None:
            return False
        if fit < self.p_fitness:
            self.p_fitness = fit
            self.p_best = self.position
        self.position_history.append(self.p_best)
        return True


    # Calculate cognitive component
    def cognitive_component(self):
        rand = np.random.rand(self.nvars)
        return self.g_learning_coef * rand * (self.p_best - self.position)
    
    # calculate social component 
    def social_component(self):        
        rand = np.random.rand(self.nvars)
        return self.p_learning_coef * rand * (self.g_best - self.position)

    # Update Velocity of particle
    def update_velocity(self, g_best):
        self.g_best = g_best
        self.velocity = self.inertia * self.velocity + self.cognitive_component() + self.social_component()
    
    # Update particle position
    def move(self):
        self.position = self.position + self.velocity * self.lr

    # Apply position limit
    def limit_position(self):
        for i in range(self.nvars):
            self.position[i] = np.minimum(np.maximum(self.position[i], self.var_min[i]), self.var_max[i])

    # Apply velocity limit
    def limit_velocity(self):
        for i in range(self.nvars):
            self.velocity[i] = np.minimum(np.maximum(self.velocity[i], self.vel_min[i]), self.vel_max[i])


    # Velocity mirror effect
    def velocity_reverse(self):
        for i in range(self.nvars):
            if self.position[i] > self.var_max[i] or self.position[i] < self.var_min[i]:
                self.velocity[i] = -self.velocity[i]


class PSO_Algorithm:
    def __init__(self, obj_func, constraints, nvars, population, birdstep, lower_bound, upper_bound, inertia_damp_mode=True, plot_result=False) -> None:
        self.population = population # Particles number
        self.birdstep = birdstep   # Maximum number of iteration
        self.nvars = nvars      # Number of decision variables

        self.lr = 0.1
        self.var_min = lower_bound   # Lower Bound of Variables
        self.var_max = upper_bound   # Upper bound of Variables

        self.inertia = 1           # Inertia weight
        self.inertia_damp = 0.99    # Inertia weight damping ratio
        
        self.g_learning_coef_start = 0.2        # Global learning coefficient
        self.g_learning_coef_end = 1.2
        self.g_learning_coef = self.g_learning_coef_start

        self.p_learning_coef_start = 1.5        # Personal learning coefficient
        self.p_learning_coef_end = 0.5
        self.p_learning_coef = self.p_learning_coef_start
        
        self.inertia_damp_mode = inertia_damp_mode # Turn on/off the adaptive inertia damping ratio

        self.g_best = np.zeros((1, self.nvars))  # Global best
        self.p_best = np.zeros((self.population, self.nvars))  # Personal best

        self.p_fitness = np.zeros(self.population)   # Personal fitness
        self.g_fitness = np.inf                         # Global fitness
        self.obj_func = obj_func

        self.constraints = constraints 

        self.particles:Particle = []  # Generate Particles
        self.g_fitness_log = []   # Fitness value log
        self.plot_result = plot_result     # plot result if True

    def init_particles(self):
        for i in range(self.population):
            
            self.particles.append(
                Particle(self)) # pass the object PSO_Algorithm itself to Particle
                #Particle(self.obj_func, self.nvars, self.inertia, self.g_learning_coef, self.p_learning_coef, self.lr, self.var_max, self.var_min))
            # print(self.particles[i].position)
            if self.particles[i].p_fitness < self.g_fitness:
                self.g_fitness = self.particles[i].p_fitness
                self.g_best = self.particles[i].p_best
    
    # method for setring attribute 
    def set_inertia_damping_mode(self, mode=True):
        self.inertia_damp_mode = mode
    
    def set_plot_mode(self,mode=True):
        self.plot_result = mode

    def set_learning_rate(self, lr):
        self.lr = lr

    def set_population(self, population:int):
        self.population = population

    def set_max_iteration(self, max_iter:int):
        self.birdstep = max_iter

    def set_inertia_damp(self, inertia_damp):
        self.inertia_damp = inertia_damp

    def update_g_learning_coef(self, generation):
        self.g_learning_coef = self.g_learning_coef_start + \
                        (self.g_learning_coef_end - self.g_learning_coef_start)*generation/self.birdstep
    
    def update_p_learning_coef(self, generation):
        self.p_learning_coef = self.p_learning_coef_start + \
                        (self.p_learning_coef_end - self.p_learning_coef_start)*generation/self.birdstep

    def move_particles(self):
        

        for particle in tqdm(self.particles):
            over_constrain_flag = False

            # Update Velocity
            particle.update_velocity(self.g_best)

            # Apply Velocity Limit
            particle.limit_velocity()

            while over_constrain_flag == False:

                # Update Position
                particle.move()

                # Velocity Mirror Effect
                particle.velocity_reverse()

                # Apply Position Limit
                particle.limit_position()


                # Evaluation and update Personal Best
                over_constrain_flag = particle.update_p_best()

            # Update Global Best
            if particle.p_fitness < self.g_fitness:
                self.g_fitness = particle.p_fitness
                self.g_best = particle.p_best

    def update(self):
        for step in range(self.birdstep):
            self.move_particles()
            self.g_fitness_log.append(self.g_fitness)

            # linearly update learning coefficients, p-->decrease, g-->increase 
            self.update_g_learning_coef(step)
            self.update_p_learning_coef(step)
            
            print(f'Iteration: {step}/{self.birdstep}   fitness {self.g_fitness}   position {self.g_best}')
            if self.inertia_damp_mode:
                self.inertia = self.inertia * self.inertia_damp

        if self.plot_result:
            self.plot_loss()
    
    def plot_particle_history(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        points, = ax.plot([], [],[], 'o', markersize=2)

        range = [a - b for a,b in zip(self.var_max, self.var_min)]
        ax.set_xlim(self.var_min[0]-0.1*range[0], self.var_max[0]+0.1*range[0])
        ax.set_ylim(self.var_min[1]-0.1*range[1], self.var_max[1]+0.1*range[1])
        ax.set_zlim(self.var_min[2]-0.1*range[2], self.var_max[2]+0.1*range[2])

        ax.set_xlabel('link 1')
        ax.set_ylabel('link 2')
        ax.set_zlabel('link 3')

        # Create Text objects for dynamic text
        text = ax.text(0.5, -0.1, 0, "", ha='center')
    

        def update(frame):
            x,y,z = [],[],[]
         

            # print(f'frame type{type(frame)}')

            for particle in self.particles:
                x.append(particle.position_history[frame][0])
                y.append(particle.position_history[frame][1])
                z.append(particle.position_history[frame][2])
            # print(frame)
            
            text.set_text(f'iteration {frame}')

            points.set_data(x,y)
            points.set_3d_properties(z)

            return points,text,
    
        num_frames = self.birdstep
        ani = FuncAnimation(fig, update, frames=num_frames, interval=50, blit=True)
        plt.show()
        plt.close(fig)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        points, = ax.plot([], [],[], 'o', markersize=2)
    
        ax.set_xlim(self.var_min[3]-0.1*range[3], self.var_max[3]+0.1*range[3])
        ax.set_ylim(self.var_min[4]-0.1*range[4], self.var_max[4]+0.1*range[4])
        ax.set_zlim(self.var_min[5]-0.1*range[5], self.var_max[5]+0.1*range[5])

        ax.set_xlabel('link 4')
        ax.set_ylabel('link 5')
        ax.set_zlabel('link 6')



        def update2(frame):
            x,y,z = [],[],[]

            # print(f'frame type{type(frame)}')

            for particle in self.particles:
                x.append(particle.position_history[frame][3])
                y.append(particle.position_history[frame][4])
                z.append(particle.position_history[frame][5])
            
            # print(x,y,z)

            points.set_data(x,y)
            points.set_3d_properties(z)

            return points,

        num_frames = self.birdstep
        ani = FuncAnimation(fig, update2, frames=num_frames, interval=50, blit=True)


        plt.show()

    def plot_loss(self):
        
        index = np.array(range(self.birdstep))
        plt.plot(index, self.g_fitness_log)
        plt.title(f'final fitness:{str(self.g_fitness)}')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()