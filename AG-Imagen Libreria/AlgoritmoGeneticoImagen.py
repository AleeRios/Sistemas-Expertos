

import numpy as np
import imageio as im
import pygad as pd
import matplotlib.pyplot as plt
import functools as fl
import operator as op

def img2chromosome(img_arr):
    """
    Represents the image as a 1D vector.
    
    img_arr: The image to be converted into a vector.
    
    Returns the vector.
    """

    return np.reshape(a=img_arr, newshape=(fl.reduce(op.mul, img_arr.shape)))

def chromosome2img(vector, shape):
    """
    Converts a 1D vector into an array.
    
    vector: The vector to be converted into an array.
    shape: The shape of the target array.
    
    Returns the array.
    """

    # Check if the vector can be reshaped according to the specified shape.
    if len(vector) != fl.reduce(op.mul, shape):
        raise ValueError("A vector of length {vector_length} into an array of shape {shape}.".format(vector_length=len(vector), shape=shape))

    return np.reshape(a=vector, newshape=shape)

def fitness_fun(solution, solution_idx):
    """
    Calculating the fitness value for a solution in the population.
    The fitness value is calculated using the sum of absolute difference between genes values in the original and reproduced chromosomes.
    
    solution: Current solution in the population to calculate its fitness.
    solution_idx: Index of the solution within the population.
    """

    fitness = np.sum(np.abs(target_chromosome-solution))

    # Negating the fitness value to make it increasing rather than decreasing.
    fitness = np.sum(target_chromosome) - fitness
    return fitness

def callback(ga_instance):
    print("Generation = {gen}".format(gen=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))

    if ga_instance.generations_completed % 500 == 0:
        plt.imsave('solution_'+str(ga_instance.generations_completed)+'.png', chromosome2img(ga_instance.best_solution()[0], target_im.shape))

# Reading target image to be reproduced using Genetic Algorithm (GA).
target_im = im.imread("C:/Users/alebe/OneDrive - Universidad Autónoma del Estado de México/Documents/Python Scripts/Sistemas Expertos/AG-Imagen Libreria/alex.jpg") # https://github.com/ahmedfgad/GARI/blob/master/fruit.jpg
target_im = np.asarray(target_im/255, dtype=np.float)

# Target image after enconding. Value encoding is used.
target_chromosome = img2chromosome(target_im)

ga_instance = pd.GA(num_generations=10000,
                       num_parents_mating=10,
                       fitness_func=fitness_fun,
                       sol_per_pop=20,
                       num_genes=target_im.size,
                       init_range_low=0.0,
                       init_range_high=1.0,
                       mutation_percent_genes=0.01,
                       mutation_type="random",
                       mutation_by_replacement=True,
                       random_mutation_min_val=0.0,
                       random_mutation_max_val=1.0,
                       callback_generation=callback)

ga_instance.run()

ga_instance.plot_result()

# Returning the details of the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

if ga_instance.best_solution_generation != -1:
    print("Best fitness value reached after {best_solution_generation} generations.".format(best_solution_generation=ga_instance.best_solution_generation))

result = chromosome2img(solution, target_im.shape)

plt.subplot(122)
plt.imshow(result)
plt.title("Resultado")
plt.subplot(121)
plt.imshow(target_im)
plt.title("Original")
plt.show()
