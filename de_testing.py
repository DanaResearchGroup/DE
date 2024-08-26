#!/usr/bin/env python3
# encoding: utf-8

import unittest
import numpy as np
from population import Population
from differential_evolution import DifferentialEvolution

class TestDifferentialEvolution(unittest.TestCase):

    def setUp(self):
        # Objective function: simple quadratic func  where we expect the minimum to be [0,0,0,0,0]
        self.objective_function = lambda x: np.sum(x ** 2)
        self.bounds1 = np.array([[-10, -10], [10, 10]])
        self.de1 = DifferentialEvolution(self.objective_function, self.bounds1, population_size=50, mutation_factor=0.8,
                                         crossover_prob=0.9, max_iter=1000)
        self.bounds2 = np.array([[-10, -10,-10,-10,-10], [10, 10, 10, 10, 10]])
        self.de2 = DifferentialEvolution(self.objective_function, self.bounds2, population_size=50, mutation_factor=0.8, crossover_prob=0.9, max_iter=1000)

    def test_mutation(self):
        # Test if mutation produces a valid mutant vector within bounds
        mutant = self.de1.mutate(0)
        self.assertEqual(len(mutant), len(self.bounds1[0]))
        self.assertTrue(np.all(mutant >= -10) and np.all(mutant <= 10))

    def test_crossover(self):
        # Test if crossover produces a valid trial vector
        target = self.de1.population.get_individuals(0)
        mutant = self.de1.mutate(0)
        trial = self.de1.crossover(target, mutant)
        self.assertEqual(len(trial), len(self.bounds1[0]))
        self.assertTrue(np.all(trial >= -10) and np.all(trial <= 10))

    def test_selection(self):
        # Test if selection updates the population correctly
        target = self.de1.population.get_individuals(0)
        mutant = self.de1.mutate(0)
        trial = self.de1.crossover(target, mutant)
        self.de1.select(0, trial)
        updated_individual = self.de1.population.get_individuals(0)
        if self.objective_function(trial) < self.objective_function(target):
            np.testing.assert_array_equal(updated_individual, trial)
        else:
            np.testing.assert_array_equal(updated_individual, target)

    def test_run(self):
        # Test the full run method
        best_solution, best_fitness = self.de1.run()
        print("best solution is:",best_solution,"best fitness is:", best_fitness)
        self.assertIsNotNone(best_solution)
        self.assertTrue(np.all(best_solution >= -10) and np.all(best_solution <= 10))
        self.assertAlmostEqual(best_fitness, 0, delta=1e-3)


if __name__ == '__main__':
    unittest.main()
