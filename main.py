import math
import time

import Reporter
import numpy as np
import random
from matplotlib import pyplot as plt
import tracemalloc
import os
import linecache
from multiprocessing import Pool

# Modify the class name to match your student number.
class r0869271:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        self.POP_SIZE = 0
        self.NUM_CITIES = 0
        self.rng = np.random.default_rng()  # TODO: do we want to choose seed?
        #self.pool = mp.Pool(mp.cpu_count())


        self.distanceMatrix = None

        # params for mutation
        self.mutation_rate = 0.20

        # params for recombination

        # params for k tournament selection
        self.k = 2 #estaba en 5 con el 2 opt
        self.lamda = 2

    def init_dependant_vars(self) -> None:
        """
        Estimate good values for the parameters that depend on the problem size.
        :return: None
        """

        if self.NUM_CITIES <= 500:

            self.POP_SIZE= 600
        else:
            self.POP_SIZE = 500
        #self.lamda = lam * self.POP_SIZE #
        #self.k = k
        #self.mutation_rate = mut

        #self.POP_SIZE = 700 #500
        #self.lamda = 2 * self.POP_SIZE  #
        #self.k= 3
        #self.mutation_rate = 0.25


    def optimize(self, filename="tour750.csv"):
        # The evolutionary algorithm's main loop
        save_history = True
        trace_memory = False
        if trace_memory:
            tracemalloc.start()

        # Read distance matrix from file.
        file = open(filename)
        self.distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()
        self.NUM_CITIES = self.distanceMatrix.shape[0]

        # get population sizes and city numbers based on the data
        self.init_dependant_vars()  #lkajflsdkjf
        population = self.initialization(self.POP_SIZE, self.NUM_CITIES)
        population = self.two_opt_sample(population, 1) #seeding

        # optimal values of TSP-29 found by external optimization tool, use for evaluation
        bestVal = 27154.488399244645
        bestVal = 27154.488399244645
        bestPath = [0, 1, 4, 7, 3, 2, 6, 8, 12, 13, 15, 23, 24, 26, 19, 25, 27, 28, 22, 21, 20, 16, 17, 18, 14, 11, 10,
                    9, 5]
        # from python_tsp.heuristics import solve_tsp_simulated_annealing
        # permutation, distance = solve_tsp_simulated_annealing(self.distanceMatrix)
        # print(distance)

        # convergence criteria

        # iteration based stopping
        max_iteration = 1.6 * 10 ** 3
        cur_iteration = 0

        # time based stopping
        max_time = 300
        start_time = time.time()

        # improvement based stopping
        # TODO: use best or mean fitness for stopping?
        bestObjective = float('NaN')  # initialization
        meanObjective= float('NaN')  # initialization
        no_improvement_counter = 0
        max_no_improvement = 60  # if for max_no_improvement the best solution has not improved, stop the algo
        relative_min_increase = 0.005 #0.01
        stop_iteration = - 1
        # save history of optimization

        means = []
        stds = []
        bests = []
        worsts = []
        times = []
        while (time.time() - start_time) < max_time and \
                (cur_iteration := cur_iteration + 1) < max_iteration and \
                True:   #no_improvement_counter < max_no_improvement


            fitness_vals = self.fitness2(population)

            last_best = bestObjective
            meanObjective = np.mean(fitness_vals[np.isfinite(fitness_vals)])
            best_index = np.argmin(fitness_vals)
            bestObjective = fitness_vals[best_index]
            bestSolution = population[best_index]

            # history values
            if save_history:
                means.append(meanObjective)
                bests.append(bestObjective)
                worsts.append(np.max(fitness_vals))
                times.append(time.time() - start_time)
                stds.append(fitness_vals.std(ddof=1))

            # check whether the best fitness value has improved since the last iteration
            # and increase or reset the counter accordingly
            #print(last_best)
            #print(bestObjective)
            if ((last_best - bestObjective) / last_best) < relative_min_increase:

                no_improvement_counter += 1
                if no_improvement_counter == max_no_improvement and stop_iteration == -1:

                    stop_iteration = cur_iteration - 1

            else:
                no_improvement_counter = 0


            if no_improvement_counter>20:
                self.lamda = 2 * self.POP_SIZE
                parent_pop = self.ktournament2(population, fitness_vals, self.lamda)  # selection
                offspring_pop = self.ox1(parent_pop)  # recombination of selected individuals
            else:
                self.lamda = 2 * self.POP_SIZE
                parent_pop = self.ktournament2(population, fitness_vals, self.lamda)  # selection
                offspring_pop = self.ox1(parent_pop)  # recombination of selected individuals

            if no_improvement_counter >15:
                self.mutation_rate= 0.35
                mutated_pop = self.insert_mutation(offspring_pop)  # mutation of offspring
                #mutated_pop = self.inversion_mutation(offspring_pop) # iversion mutation
            else:
                self.mutation_rate = 0.20
                mutated_pop = self.insert_mutation(offspring_pop)  # mutation of offspring
                #mutated_pop = self.inversion_mutation(offspring_pop) # iversion mutation

            #two opt only on population
            population = self.two_opt_sample(population, 3)

            #CROWDING
            #new_population = self.crowding(population, mutated_pop)


            #NEW
            combined_pop = np.vstack([population, mutated_pop])

            #ANTES HABÍA MÁS OFFSPRING
            #combined_pop = mutated_pop


            ## LOCAL SEARCH OPERATOR SWAP
            #combined_pop= self.localsearch_sample(combined_pop)

            # LOCAL SEARCH OPERATOR TWO OPT on the combined population
            #combined_pop = self.two_opt_sample(combined_pop,3)

            fitness_vals = self.fitness2(combined_pop)
            # elimination k tournamnet with replacement
            #population = self.ktournament2(combined_pop, fitness_vals, self.POP_SIZE)
            # elimination ktournament without replacement
            population = self.elimination2(combined_pop, fitness_vals, self.POP_SIZE)
            # elimination elitism
            #population = self.elimination_elitism(combined_pop, fitness_vals)

            # Your code here.
            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution
            #    with city numbering starting from 0
            #timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            #if timeLeft < 0:
                 #break

            if (cur_iteration % 20) == 0:
                print(f"{cur_iteration=}")
                print(f"{meanObjective=}")
                print(f"{bestObjective=}")
        print(f"pop size= {self.POP_SIZE}, lambda ={self.lamda}, k= {self.k}, mutation rate={self.mutation_rate}.")
        print(f"Optimization ended after {cur_iteration} iterations.")
        print(f"It took {time.time() - start_time} seconds")
        print(f"The stopping criterion was reached after {stop_iteration} iterations in")
        if save_history: print(f"Stopped at value {bests[stop_iteration]}")
        print(f"Best fitness value {bestObjective}.")
        if trace_memory:
            print("Current: %d, Peak %d" % tracemalloc.get_traced_memory())
            snapshot = tracemalloc.take_snapshot()
            display_top(snapshot)
        #if save_history:
            #plot_iterations(means, bests, times, stds, worsts, stop_iteration) Plot iterations
        return 0

    def initialization(self, pop_size: int, num_cities: int) -> np.ndarray:
        """
        Initializes a population of pop_size that contains solutions of size num_cities.
        A path representation is used to represent solutions to a TSP with num_cities cities.
        Each member of the population consists of a random permutation of the integers from 0 to num_cities(exclusive).
        The members are guaranteed to always start with the city with the highest ID
        so that there is a unique representation for each cycle.
        :param pop_size: the number of candidates
        :param num_cities: the length of each candidate
        :return: a numpy array of shape (pop_size, num_cities)
        """
        num_cities = num_cities - 1
        raw_candidates = np.arange(pop_size * num_cities) % num_cities
        reshaped_candidates = raw_candidates.reshape((pop_size, num_cities))
        permutated_candidates = self.rng.permuted(reshaped_candidates, axis=1)
        permutated_candidates = np.hstack([np.full((pop_size, 1), num_cities), permutated_candidates])
        # LOCAL SEARCH swap
        #permutated_candidates =self.localsearch_total(permutated_candidates)
        # LOCAL SEARCH TWO OPT
        #optimized, fitness=self.two_opt(permutated_candidates[0])
        #print(f"optimized and fitness de k opt{optimized, fitness}")
        #permutated_candidates[0]= optimized
        return permutated_candidates

    def initializationReplaceCandidate(self, num_cities):
        candidate = np.arange(num_cities)
        permutated_candidate = self.rng.permuted(candidate)
        return permutated_candidate

    def fitness(self, population: np.ndarray) -> np.ndarray:
        """
        Evaluates the fitness of the given population, assuming a path representation of the population.
        The distance matrix of the class is used to calculate the distance of the TSP path for each solution
        :param population: The population of the candidates whose fitness should be evaluated
        :return: a numpy array of size (population.shape[0],1) that contains the fitness value of each candidate
        """
        # TODO: vectorize, probably too inefficient for large populations
        fitness_vals = np.zeros(population.shape[0])
        for i, candidate in enumerate(population):
            fitness_vals[i] = self.distanceMatrix[candidate[:-1], candidate[1:]].sum()
            fitness_vals[i] += self.distanceMatrix[candidate[-1], candidate[0]]
        return fitness_vals

    def fitness2(self, population):
        population = np.hstack([population, population[:, [0]]])  # append the first city to the end
        cost_matrix = self.distanceMatrix[population[:, :-1], population[:, 1:]]
        cost_matrix[cost_matrix == np.inf] = 100000000
        return cost_matrix.sum(axis=1)

    def fitness2_candidate(self, population):
        population = np.hstack([population, population[0]])  # append the first city to the end
        cost_matrix = self.distanceMatrix[population[:-1], population[1:]]
        cost_matrix[cost_matrix == np.inf] = 100000000
        return cost_matrix.sum()

    def fitness2_candidate_kopt(self, population): #YA NO LO USE
        population = np.hstack([population, population[0]])  # append the first city to the end
        cost_matrix = self.distanceMatrix[population[:-1], population[1:]]
        cost_matrix[cost_matrix == np.inf] = 100000000
        return cost_matrix

    def insert_mutation(self, population: np.ndarray) -> np.ndarray:
        """
        Mutates each candidate of the given population with the probability self.mutation rate.
        In the candidates that are chosen to be mutated a random element is taken and inserted into a different position
        :param population: A population of candidates
        :return: the population with mutated members
        """
        pop_size, num_cities = population.shape
        candidate_indices = self.rng.random(pop_size) < self.mutation_rate
        candidates = np.nonzero(candidate_indices)[0]
        # 1st column of elements decides from where the element should be taken, second one where it should be inserted
        elements = self.rng.integers(1, num_cities,
                                     (len(candidate_indices), 2))  # element 0 should always stay in place
        new_pop = population.copy()
        for candidate, (source, target) in zip(candidates, elements):
            pop_cand = population[candidate]  # population candidate that is mutated in this iteration
            # Two different cases depending on if the insertion site precedes
            # or antecedes the place from where it is taken
            if source < target:
                new_pop[candidate] = np.concatenate(
                    [pop_cand[:source], pop_cand[source + 1:target],
                     [pop_cand[source]],
                     pop_cand[target:]])
            elif target < source:
                new_pop[candidate] = np.concatenate(
                    [pop_cand[:target],
                     [pop_cand[source]],
                     pop_cand[target:source],
                     pop_cand[source + 1:]])
        return new_pop

    def inversion_mutation(self, population: np.ndarray) -> np.ndarray:
        """
        Mutates each candidate of the given population with the probability self.mutation rate.
        In the candidates that are chosen to be mutated a random section is taken and inverted
        :param population: A population of candidates
        :return: the population with mutated members
        """
        pop_size, num_cities = population.shape
        candidate_indices = self.rng.random(pop_size) < self.mutation_rate
        candidates = np.nonzero(candidate_indices)[0]
        # 1st column of elements decides from where the element should be taken, second one where it should be inserted
        elements = self.rng.integers(1, num_cities,
                                     (len(candidate_indices), 2))  # element 0 should always stay in place
        new_pop = population.copy()
        for candidate, (a, b) in zip(candidates, elements):
            pop_cand = population[candidate]  # population candidate that is mutated in this iteration

            if a > b:  # sort the cut off points
                a, b = b, a
            population[candidate][a:b] = population[candidate][a:b][::-1]

        return population

    def inversion_mutation_crowding(self, candidate: np.ndarray) -> np.ndarray:
        """
        Mutates each candidate of the given population with the probability self.mutation rate.
        In the candidates that are chosen to be mutated a random section is taken and inverted
        :param candidate to be inverted
        :return: candidate with mutated members
        """
        pop_size, num_cities = candidate.shape

        a = self.rng.integers(1, num_cities)
        b= self.rng.integers(1, num_cities)
        if a > b:  # sort the cut off points
            a, b = b, a
        candidate[a:b]= candidate[a:b][::-1]
        return candidate


    def two_opt_sample(self, population, sample, improvement_thresh=0.01):
        #se le da a la función una population
        pop_size, num_cities = population.shape
        #sample = 3
        candidates = self.rng.integers(0, pop_size, sample)
        for c in range(sample):
            candidate = population[candidates, :][c]
            best_path = candidate.copy()
            best_fitness = self.fitness2_candidate(candidate)
            improvement = 1
            while improvement > improvement_thresh:
                previous_best = best_fitness
                for swap_first in range(1, num_cities - 2):
                    for swap_last in range(swap_first + 1, num_cities - 1):
                        p1 = best_path[swap_first - 1]
                        p2 = best_path[swap_first]
                        p3 = best_path[swap_last]
                        p4 = best_path[swap_last + 1]
                        before = self.distanceMatrix[p1][p2] + self.distanceMatrix[p3][p4]
                        after = self.distanceMatrix[p1][p3] + self.distanceMatrix[p2][p4]
                        if after < before:
                            best_path = np.concatenate((best_path[0:swap_first],
                                                        best_path[swap_last:-len(best_path) + swap_first - 1:-1],
                                                        best_path[swap_last + 1:len(best_path)]))

                            best_fitness = self.fitness2_candidate(best_path)

                improvement = 1 - best_fitness / previous_best
            population[c, :] = best_path.copy()
        return population

    def localsearch_sample(self, population):
        #aplicar local search operator at certain depth to a sample population
        pop_size, num_cities = population.shape
        sample = 5
        candidates = self.rng.integers(0, pop_size, sample)
        depth = 60 #TENGO QUE PONERLO DE ACUERDO AL NUMERO MÁXIMO DE CIUDADES
        for c in range(sample):
            candidate = population[candidates, :][c]
            best_fitness = self.fitness2_candidate(candidate)
            best_path = candidate.copy()
            # HACER RANDOM EL i??
            for i in range(depth):
                for j in range(i + 1, depth):
                    neighbour = candidate.copy()
                    neighbour[i], neighbour[j] = candidate[j], candidate[i]
                    fitness = self.fitness2_candidate(neighbour)
                    if fitness < best_fitness:
                        best_path = neighbour.copy()
                        best_fitness = fitness
                    neighbour[i] = candidate[i]
                    neighbour[j] = candidate[j]
            population[candidates[c], :] = best_path.copy()
        return population


    def localsearch_total (self, population):

        num_cities= population.shape[1]
        sample=5 #how many candidates to apply the local search operator
        for c in range(sample):
            candidate = population[c, :]
            best_fitness = self.fitness2_candidate(candidate)
            best_path = candidate.copy()
            for i in range(candidate.shape[0]):
                for j in range(i+1, candidate.shape[0]):
                    neighbour = candidate.copy()
                    neighbour[i], neighbour[j] =candidate[j], candidate[i]
                    fitness = self.fitness2_candidate(neighbour)
                    if fitness < best_fitness:
                        best_path = neighbour.copy()
                        best_fitness = fitness
                    neighbour[i] = candidate[i]
                    neighbour[j] = candidate[j]
            population[c, : ] = best_path.copy()
        return population


    def ktournament(self, population, fitness_vals, num_offspring):
        """
        Perform k-tournament selection to select pairs of parents
        """
        num_cities = population.shape[1]
        selected = np.zeros((num_offspring, num_cities), dtype=np.int32)
        for ii in range(num_offspring):  # self.lambda
            # ri = random.choices(range(np.size(population,0)), k = self.k)
            # ri = random.sample(range(np.size(population,0)), self.k) #without replacement

            ri = random.choices(range(np.size(population, 0)), k=self.k)  # with replacement
            # print(ri)
            ca = np.argmin(fitness_vals[ri])  # ca ist entweder 1 oder 2
            # print(ri[ca])
            # print(population[(ri[ca]), :])
            # Einspeichern
            selected[ii, :] = population[(ri[ca]), :]
        return selected


    def elimination2(self, population, fitness_vals, pop_size):
        """
        Perform k-tournament selection to select survivors without replacement
        """
        pop_size, num_cities = population.shape
        # chooses num_chosen groups of k elements from the population (without replacement)
        x=int(pop_size/self.POP_SIZE)
        candidates = self.rng.choice(range(np.size(population, 0)), (x, self.POP_SIZE), replace=False) #estaba en sel.k
        # chooses the best candidate from each group
        winner_indices = np.argmin(fitness_vals[candidates], axis=0)
        best_candidate_indices = candidates[winner_indices, np.arange(self.POP_SIZE)]
        return population[best_candidate_indices]

    def ktournament2(self, population, fitness_vals, num_offspring):
        """
        Perform k-tournament selection to select pairs of parents with replacement
        """
        pop_size, num_cities = population.shape
        # chooses num_chosen groups of k elements from the population (with replacement)

        candidates = self.rng.integers(0, pop_size, (self.k, num_offspring))
        # chooses the best candidate from each group
        winner_indices = np.argmin(fitness_vals[candidates], axis=0)
        best_candidate_indices = candidates[winner_indices, np.arange(num_offspring)]
        return population[best_candidate_indices]

    def crowding(self, population, offspring):
        pop_size, num_cities = population.shape #PORUQE SE LE CAMBIO AQUÍ A OFFSPRING NO SIRVE
        fitnessValsPopulation = self.fitness2(population)
        #candidates = self.rng.integers(0, pop_size, (2, 1)) #Crowding factor =2 ESTA MAL AQUI
        for i in range(pop_size):
            candidates = self.rng.integers(0, pop_size, (2, 1))
            fit_offspring = self.fitness2_candidate(offspring[i])
            #TODO Crowding factor =2
            caca2 = np.absolute(fitnessValsPopulation[candidates] - fit_offspring) #PROBLEMA AQUI DE OUT OF BOUNDS
            looser = np.argmin(caca2)
            #new_candidate = self.initializationReplaceCandidate(self.NUM_CITIES) #replace looser with a random
            # population = np.delete(population, (candidates[looser]), axis=0) #delete looser
            toInvert=population[candidates[looser]].copy() #candidate to mutate

            new_candidate = self.insert_mutation(toInvert)
            #new_candidate =self.inversion_mutation_crowding(toInvert) #inversion mutation
            population[candidates[looser]]= new_candidate

        return population

    def elimination_elitism(self, population, fitness_values):
        perm = np.argsort(fitness_values)
        survivors = population[perm[1:self.POP_SIZE], :]
        return survivors

    ####################################################################################################################
    #   Recombination operators                                                                                        #
    ####################################################################################################################
    def ox1(self, population: np.ndarray) -> np.ndarray:
        """
        Use OX or order crossover operator to generate offspring from the given parent population.
        Two cut off points are chosen and the segment inbetween is kept for each offspring.
        The remaining cities are filled in according to the order they are in the other parent.
        :param population: the parent population
        :return: offspring population with the same size, rounded down to an even number
        """
        pop_size, num_cities = population.shape
        # since each individual needs two parents, for uneven population sizes the last one is ignored
        if (pop_size % 2) == 1:
            pop_size = pop_size - 1
        cut_off_points = self.rng.integers(0, num_cities, (pop_size // 2, 2))
        offspring = np.empty((pop_size, num_cities), dtype=np.int32)
        for i, (a, b) in enumerate(cut_off_points):
            parent1 = population[2 * i]
            parent2 = population[2 * i + 1]
            if a > b:  # sort the cut off points
                a, b = b, a
            # boolean arrays where the values in the cutoff segments are flagged to prevent duplication
            # indices correspond to city numbers, not positions in the permutation
            flag_values_1 = np.zeros(num_cities)
            flag_values_2 = np.zeros(num_cities)
            for i_hole in range(a, b):  # iteration over the cutoff segment
                flag_values_1[parent1[i_hole]] = True
                flag_values_2[parent2[i_hole]] = True
                offspring[2 * i, i_hole] = parent1[i_hole]  # copy the cutoff segments into the offspring
                offspring[2 * i + 1, i_hole] = parent2[i_hole]

            # copy the remaining values
            ind_offspring1 = b  # start after the second cutoff point
            ind_offspring2 = b
            for parent_ind in range(num_cities):
                parent_ind = (b + parent_ind) % num_cities
                # if values were not already in the cutoff segment,
                # copy the value from the other parent and increase index
                # otherwise skip the value from the parent
                if not flag_values_1[parent2[parent_ind]]:
                    offspring[2 * i, ind_offspring1] = parent2[parent_ind]
                    ind_offspring1 = (ind_offspring1 + 1) % num_cities
                if not flag_values_2[parent1[parent_ind]]:
                    offspring[2 * i + 1, ind_offspring2] = parent1[parent_ind]
                    ind_offspring2 = (ind_offspring2 + 1) % num_cities
        return offspring


def plot_iterations(means, bests, times, stds, worsts, stop_iteration):
    plt.figure(1)
    plt.title("Overview of an optimization process")
    plt.xlabel("Iterations")
    plt.ylabel("Fitness value")
    plt.scatter(np.arange(len(bests)), bests, color="red", label="Best value", )
    plt.scatter(np.arange(len(bests)), means, color="blue", label="Mean value")
    # plt.scatter(np.arange(len(bests)), worsts, color="green", label="Worst value")
    #plt.hlines(27154.488399244645, 0, len(bests), label="Global minimum", ls="dotted") # 27154.488399244645 (29 cities)
    plt.hlines(272865, 0, len(bests), label="Greedy heuristic", ls="dashed") # 30350 (29 cities) 272865  (100 cities)
    #49889(250 cities) 122355 (500 cities) 119156 (750cities) 226541(1000 cities)
    plt.scatter(stop_iteration, bests[stop_iteration], marker="*", label="Stopping criterion", s=150, c="yellow",
                edgecolors="black")

    # variation of the fitness values plotted as shad
    num_stds = 2
    means = np.array(means)
    stds = np.array(stds)
    plt.fill_between(np.arange(len(bests)), means - num_stds * stds, means + num_stds * stds, alpha=0.3, zorder=-100)
    plt.ylim(200000, 300000)

    # second x axis on top that shows wall time
    avg_time = times[-1] / len(times)
    secax = plt.gca().secondary_xaxis('top', functions=(lambda x: x * avg_time, lambda y: y / avg_time))
    secax.set_xlabel('Wall time [s]')

    plt.tight_layout()
    plt.legend()
    plt.show()


def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
        ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


if __name__ == '__main__':
    b = r0869271()
    a = b.optimize()