import numpy as np
import matplotlib.pyplot as plt
from differential_evolution import DifferentialEvolution

def objective_function(x):
    return np.sum(x ** 2, axis=0)

# Visualization function to plot population and contours of the objective function
def plot_population(de, iteration):
    plt.figure(figsize=(8, 6))

    # Create a grid to evaluate the objective function
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = objective_function(np.array([X, Y]))

    # Plot the contours of the objective function
    plt.contour(X, Y, Z, levels=50, cmap='viridis')

    # Plot the population
    population = np.array([de.population.get_individuals(i) for i in range(de.population_size)])
    plt.scatter(population[:, 0], population[:, 1], color='red')

    # Mark the best solution found so far
    if de.best_solution is not None:
        plt.scatter(de.best_solution[0], de.best_solution[1], color='blue', marker='*', s=100)

    plt.title(f"Iteration {iteration}")
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


# Custom run function that shows the optimization process
def run_with_visualization(de):
    for iteration in range(de.max_iter):
        print(f"Iteration: {iteration + 1}")
        for i in range(de.population_size):
            target = de.population.get_individuals(i)
            mutant = de.mutate(i)
            trial = de.crossover(target, mutant)
            de.select(i, trial)

        plot_population(de, iteration + 1)

    print("Final Best Solution:", de.best_solution)
    print("Final Best Fitness:", de.best_fitness)
    return de.best_solution, de.best_fitness


# Setup and run the Differential Evolution with visualization
bounds = np.array([[-10, -10], [10, 10]])  # 2D problem
de = DifferentialEvolution(objective_function, bounds, population_size=20, mutation_factor=0.8, crossover_prob=0.7,
                           max_iter=10)

best_solution, best_fitness = run_with_visualization(de)
