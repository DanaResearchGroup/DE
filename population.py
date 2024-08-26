import numpy as np


class Population:
    def __init__(self, size, bounds):
        self.size = size
        self.bounds = bounds
        self.dim = len(bounds[0])
        self.individuals = self._initialize_population()

    def _initialize_population(self):
        # Initialize population with random solutions within bounds
        return [np.random.uniform(low=self.bounds[0], high=self.bounds[1], size=len(self.bounds[0])) for _ in range(self.size)]

    def get_individuals(self, index):
        return self.individuals[index]

    def update_individual(self, index, new_individual):
        self.individuals[index] = new_individual

