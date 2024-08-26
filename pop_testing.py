#!/usr/bin/env python3
# encoding: utf-8


import unittest
import numpy as np
from population import Population


class TestPopulation(unittest.TestCase):

    def setUp(self):
        # Set up initial conditions for the tests
        self.population1 = Population(size=10, bounds=[(-10, -10), (10, 10)])
        self.population2 = Population(size=13, bounds=[(-10,-10,-10), (10,10,10)])

    def test_population_size(self):
        # Test if the population size is correct
        self.assertEqual(len(self.population1.individuals), 10)
        self.assertEqual(len(self.population2.individuals), 13)

    def test_individual_within_bounds(self):
        # Test if all individuals are within the specified bounds
        for individual in self.population1.individuals:
            self.assertTrue(np.all(individual >= -10))
            self.assertTrue(np.all(individual <= 10))

    def test_get_individual(self):
        # Test if getting an individual works correctly
        individual = self.population1.get_individuals(0)
        self.assertEqual(len(individual), 2)  # Check if individual has correct dimension
        individual = self.population2.get_individuals(0)
        self.assertEqual(len(individual), 3)  # Check if individual has correct dimension

    def test_update_individual(self):
        # Test if updating an individual works correctly
        new_individual = np.array([0, 0])
        print("original individual:", self.population1.get_individuals(0))
        self.population1.update_individual(0, new_individual)
        updated_individual = self.population1.get_individuals(0)
        np.testing.assert_array_equal(updated_individual, new_individual)  # Assert the individual was updated correctly
        print("new individual:",new_individual,"should equal updated individual:", updated_individual)

    def test_individual_dimensions(self):
        # Test if individuals have the correct number of dimensions
        for individual in self.population1.individuals:
            self.assertEqual(len(individual), 2)

        for i in self.population2.individuals:
            self.assertEqual(len(i),3)

if __name__ == '__main__':
    unittest.main()


