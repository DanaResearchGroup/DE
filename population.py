import numpy as np


class Population:
    """
    Class Particle for swarm optimization.
    Args:
        size (float, int): the size of the population (amount of individuals in the population).
        bounds (np.array[float, int, np.float, np.int]): the lower and upper limit of the search
        range. Takes a two dimension array, first vector being the lower limit of each dimension
        and the second vector- the upper limit of each dimension. Both vectors have to be the
        same length and equal to the dimensions of the given optimization problem.
    """
    def __init__(self, size, bounds):
        self.size = size
        self.bounds = bounds
        self.dim = len(bounds[0])
        self.individuals = self._initialize_population()

    def _initialize_population(self):
        """
        Func for initializing individuals of a given population size within given bounds.
        Args:
            self
        """
        return [np.random.uniform(low=self.bounds[0], high=self.bounds[1], size=len(self.bounds[0])) for _ in range(self.size)]

    def get_individuals(self, index):
        """
        Func that enables the calling of an individual from the population.
        Args:
            self, index
        """
        return self.individuals[index]

    def update_individual(self, index, new_individual):
        """
        Func for updating individuals due to mutation
        Args:
            self, index, new_individual
            This function replaces or updates an individual with a more improved fitness
            due to mutation.
        """
        self.individuals[index] = new_individual

