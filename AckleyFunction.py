import numpy as np
import math

# Base class for real-valued optimization problems
class RealValuedOptimizationProblem:
    '''
    topic 5.1: Write a class which represents real-valued optimisation problems of different numbers
    of dimensions.
    '''
    def __init__(self, dimensions):
        self.dimensions = dimensions
        
    def evaluate(self, solution):
        """
        Evaluate the solution. This should be overridden by derived classes.
        """
        raise NotImplementedError("This method should be implemented by derived classes.")
        
    def get_dimensions(self):
        return self.dimensions

# Derived class implementing the Ackley function
class AckleyFunction(RealValuedOptimizationProblem):
    '''
    topic 5.1: Write a derived class that implements the Ackley function with variable
    dimension and boundaries from Lecture 4, which is a benchmark problem for optimisers like PSO
    '''
    def __init__(self, dimensions):
        super().__init__(dimensions)
        self.a = 20
        self.b = 0.2
        self.c = 2 * math.pi
        
    def evaluate(self, solution):
        if len(solution) != self.dimensions:
            raise ValueError("The solution size does not match the problem dimensions.")
            
        sum1 = sum([x**2 for x in solution])
        sum2 = sum([math.cos(self.c * x) for x in solution])
        
        term1 = -self.a * math.exp(-self.b * math.sqrt(sum1 / self.dimensions))
        term2 = -math.exp(sum2 / self.dimensions)
        
        return term1 + term2 + self.a + math.exp(1)


class Particle:
    '''
    ramdomly initialize the position and velocity of the particle
    this is what topic 5.3 asks us to do
    '''
    # Initialize a particle with random position and velocity
    def __init__(self, dimensions, bounds):
        self.position = np.array([np.random.uniform(bounds[0], bounds[1]) for _ in range(dimensions)])
        self.velocity = np.array([np.random.uniform(-abs(bounds[1] - bounds[0]), abs(bounds[1] - bounds[0])) for _ in range(dimensions)])
        self.pbest_position = np.copy(self.position)
        self.pbest_value = float('inf')

class PSOOptimizer:
    '''
    for each parameter below, write a very brief comment explaining what it is
    '''
    def __init__(self, objective_function, dimensions,
                bounds=(-32.768, 32.768), num_particles=30, w=0.5, c1=1.5, c2=1.5, max_iterations=1000):
        # Objective function to be optimized
        self.objective_function = objective_function
        # Number of dimensions
        self.dimensions = dimensions
        # Bounds for each dimension, any particle outside these bounds will be clipped
        self.bounds = bounds
        # Number of particles in the swarm
        self.num_particles = num_particles
        # Inertia weight
        self.w = w
        # acceleration coefficients
        self.c1 = c1
        self.c2 = c2
        # Maximum number of iterations
        self.max_iterations = max_iterations
        # random initialize the global best position with no prior knowledge about the global minimum
        # this is done for each dimension
        self.gbest_position = np.array([np.random.uniform(bounds[0], bounds[1]) for _ in range(dimensions)])
        # initialize the global best value of the objective function
        self.gbest_value = float('inf')
        # initialize the particles
        self.particles = [Particle(dimensions, bounds) for _ in range(num_particles)]
        
    def evaluate(self, particle):
        return self.objective_function(particle.position)
    
    def update_pbest_gbest(self):
        '''
        function to update the personal best and global best positions and values
        '''
        for particle in self.particles:
            fitness = self.evaluate(particle)
            if fitness < particle.pbest_value:
                particle.pbest_value = fitness
                particle.pbest_position = particle.position
                
            if fitness < self.gbest_value:
                self.gbest_value = fitness
                self.gbest_position = particle.position
    
    def update_velocities_positions(self):
        '''
        function to update the velocities and positions of the particles based on the PSO algorithm
        this is where the PSO algorithm is implemented
        also where the randomization of the velocity and position comes into play
        '''
        for particle in self.particles:
            inertia = self.w * particle.velocity
            personal_attraction = self.c1 * np.random.random() * (particle.pbest_position - particle.position)
            social_attraction = self.c2 * np.random.random() * (self.gbest_position - particle.position)
            
            particle.velocity = inertia + personal_attraction + social_attraction
            particle.position += particle.velocity
            
            # Clip position values to be within bounds
            particle.position = np.clip(particle.position, self.bounds[0], self.bounds[1])
    
    def optimize(self):
        '''
        function to run the PSO algorithm for a number of iterations
        '''
        for _ in range(self.max_iterations):
            self.update_pbest_gbest()
            self.update_velocities_positions()
            
        return self.gbest_position, self.gbest_value

# For demonstration purposes, you can optimize the Ackley function using the PSOOptimizer
ackley_function = AckleyFunction(2)
pso_optimizer = PSOOptimizer(ackley_function.evaluate, dimensions=2)
best_position, best_value = pso_optimizer.optimize()
print(best_position, best_value)
