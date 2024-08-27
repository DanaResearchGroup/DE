from population import Population
import numpy as np

class DifferentialEvolution:
    """
    Class DifferentialEvolution for optimization.
        Args:
        objective_function(np.array): A vector of input variables for the function. The length of this array should match
                                    the number of dimensions in the optimization problem. Each element in the array
                                    represents a dimension value for which the objective function will be evaluated.
        bounds (np.array[float, int, np.float, np.int]): the lower and upper limit of the search
                range. Takes a two dimension array, first vector being the lower limit of each dimension
                and the second vector- the upper limit of each dimension. Both vectors have to be the
                same length and equal to the dimensions of the given optimization problem.
        population_size (int): the size of the population (amount of individuals in the population).
                                Default is set at 50. For bigger, more complex problems, we might want to increase max
                                iteration and/or population size.
        mutation_factor (float): the mutation factor is a number between 0 and 2 and is used to create a mutant vector
                                in a stochastic manner.
        crossover_prob (float): The crossover probability is a value between 0 and 1 that determines how likely it is
                                for each dimension of the trial vector to adopt values from the mutant vector rather
                                than the original vector. For each dimension, there is a chance equal to the crossover
                                probability that the value from the mutant vector will replace the corresponding value
                                in the original vector. This process helps create a new trial vector that combines
                                elements of both vectors. Default is set at 0.7.
        max_iter (int): the max number of iterations for the optimization process. Default is 1000. For bigger, more
                        complex problems, we might want to increase max iteration and/or population size.
    """
    def __init__(self, objective_function, bounds, population_size=50, mutation_factor=0.8, crossover_prob=0.7, max_iter=1000):
        self.objective_function = objective_function
        self.bounds = bounds
        self.dim = len(bounds[0])
        self.population_size = population_size
        self.mutation_factor = mutation_factor
        self.crossover_prob = crossover_prob
        self.max_iter = max_iter
        self.population = Population(population_size, bounds)
        self.best_solution = None
        self.best_fitness = float('inf')

    def mutate(self, idx):
        """
        Creates a mutant vector for each individual in the population.
        Args:
            self,idx
            The way this method works is it randomly chooses three individuals from the population (not including the
            current individual). It then creates a mutant vector using the following formula:
            mutant = a + mutation_factor*(b-c), where a,b,c are the three randomly chosen individuals and the
            mutation_factor is provided. If the mutant vector has any values out of the given bounds, the bounds
            themselves replace those values.
        """
        candidates = [i for i in range(self.population_size) if i != idx]
        a, b, c = np.random.choice(candidates, 3, replace=False)
        x_a, x_b, x_c = self.population.get_individuals(a), self.population.get_individuals(b), self.population.get_individuals(c)
        mutant = x_a + self.mutation_factor * (x_b - x_c)
        bounds_array = np.array(self.bounds)
        for i in range(len(mutant)):
            if mutant[i] < bounds_array[0, i]:
                mutant[i] = bounds_array[0, i]
            elif mutant[i] > bounds_array[1, i]:
                mutant[i] = bounds_array[1, i]
        return mutant

    def crossover(self, target, mutant):
        """
        Creates a trial vector for each individual.
        Args:
            self, target, mutant
            The trial vector is a combination of the original vector and the mutant vector created by the mutate class.
            For each dimension, there is a chance equal to the crossover probability that the value from the mutant
            vector will replace the corresponding value in the original vector. This process helps create a new trial
            vector that combines elements of both vectors.
        """
        crossover_mask = np.random.rand(self.dim) < self.crossover_prob
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def select(self, idx, trial):
        """
        Selects the higher vector with a higher fitness between the trial vector created in the crossover method and the
        original vector for each individual.
        Args:
            self, idx, trial
            This method evaluates the trial and target (original) fitness for the given objective function and selects
            the vector with the higher fitness. In addition, it updates the overall best fitness and best solution.
        """
        target_fitness = self.objective_function(self.population.get_individuals(idx))
        trial_fitness = self.objective_function(trial)
        if trial_fitness < target_fitness:
            self.population.update_individual(idx, trial)
        if trial_fitness < self.best_fitness:
            self.best_fitness = trial_fitness
            self.best_solution = trial

    def run(self):
        """
        Runs the DE optimization while utilizing all other methods prior.
        Args:
            self
            Optimizes the given random population for the given max iterations and returns the best solution and
            best fitness.
        """
        for iteration in range(self.max_iter):
            for i in range(self.population_size):
                target = self.population.get_individuals(i)
                mutant = self.mutate(i)
                trial = self.crossover(target, mutant)
                self.select(i, trial)
        return self.best_solution, self.best_fitness


