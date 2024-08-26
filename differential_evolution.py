from population import Population
import numpy as np

class DifferentialEvolution:
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
        crossover_mask = np.random.rand(self.dim) < self.crossover_prob
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def select(self, idx, trial):
        target_fitness = self.objective_function(self.population.get_individuals(idx))
        trial_fitness = self.objective_function(trial)
        if trial_fitness < target_fitness:
            self.population.update_individual(idx, trial)
        if trial_fitness < self.best_fitness:
            self.best_fitness = trial_fitness
            self.best_solution = trial

    def run(self):
        for iteration in range(self.max_iter):
            for i in range(self.population_size):
                target = self.population.get_individuals(i)
                mutant = self.mutate(i)
                trial = self.crossover(target, mutant)
                self.select(i, trial)
        return self.best_solution, self.best_fitness


