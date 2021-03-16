import random
import copy
import numpy as np
from scipy import spatial
import time
import preprocessing


def generate_similarity_matrix_for_evolutionary_algorithm(sentences_as_embeddings):
    number_of_sentences = len(sentences_as_embeddings)
    similarity_matrix = np.zeros([number_of_sentences, number_of_sentences])
    for i, row_embedding in enumerate(sentences_as_embeddings):
        for j, column_embedding in enumerate(sentences_as_embeddings):
            similarity_matrix[i][j] = 1 - spatial.distance.cosine(row_embedding, column_embedding)
    for i in range(number_of_sentences):
        for j in range(number_of_sentences):
            if j <= i:
                similarity_matrix[i][j] = 0
    return similarity_matrix


def generate_individual(text_size, summary_size):
    individual = np.zeros(text_size, dtype=bool)
    bits = random.sample(range(0, text_size), summary_size)
    for i in bits:
        individual[i] = True
    return individual.tolist()


def fitness(individual, similarity_matrix, summary_size, a=0.34, b=0.33, c=0.33):
    return (a * cohesion(individual, similarity_matrix, summary_size) + b * readability(individual, similarity_matrix) + c * sentence_position(individual)) / (a + b + c)


def topological_sort_util(current_node, stack, visited, adjacency_matrix):
    visited[current_node] = True
    for neighbor in adjacency_matrix[current_node]:
        neighbor = neighbor[0]
        if not visited[neighbor]:
            topological_sort_util(neighbor, stack, visited, adjacency_matrix)
    stack.append(current_node)


def longest_path(adjacency_matrix):
    stack = []
    visited = [False for _ in range(len(adjacency_matrix))]
    for i in range(len(adjacency_matrix)):
        if not visited[i]:
            topological_sort_util(i, stack, visited, adjacency_matrix)

    dist = [np.NINF for _ in range(len(adjacency_matrix))]
    dist[0] = 0
    while len(stack) > 0:
        current = stack[-1]
        del stack[-1]
        if dist[current] != np.NINF:
            for neighbor in adjacency_matrix[current]:
                cost = neighbor[1]
                neighbor = neighbor[0]
                if dist[neighbor] < dist[current] + cost:
                    dist[neighbor] = dist[current] + cost
    return dist


def max_weight_dag(similarity_matrix):
    adjacency_matrix = [[] for _ in range(len(similarity_matrix))]

    for i in range(len(similarity_matrix)):
        for j in range(len(similarity_matrix)):
            if similarity_matrix[i][j] > 0:
                adjacency_matrix[i].append([j, similarity_matrix[i][j]])

    return longest_path(adjacency_matrix)[len(similarity_matrix) - 1]


def readability(individual, similarity_matrix):
    indices = [i for i in range(len(individual)) if individual[i]]
    readability_sum = 0
    for i in range(len(indices) - 1):
        readability_sum += similarity_matrix[indices[i]][indices[i + 1]]
    max_readability = max_weight_dag(similarity_matrix)
    return readability_sum / max_readability


def sentence_position(individual):
    indices = [i + 1 for i in range(len(individual)) if individual[i]]
    result = 0
    for i in indices:
        result += np.sqrt(1 / i)
    size = len(indices)
    normalisation_factor = 0
    for i in range(size):
        factor = i + 1
        normalisation_factor += np.sqrt(1 / factor)
    return result / normalisation_factor


def cohesion(individual, similarity_matrix, summary_size):
    cohesion_sum = 0
    maxi = -1
    for i in range(len(individual)):
        for j in range(len(individual)):
            if individual[i] and individual[j]:
                cohesion_sum += similarity_matrix[i][j]
            if similarity_matrix[i][j] > maxi:
                maxi = similarity_matrix[i][j]
    cohesion_not_normalized = 2 * cohesion_sum / ((summary_size - 1) * summary_size)
    cohesion_normalized = np.log10(9 * cohesion_not_normalized + 1) / np.log10(9 * maxi + 1)
    return cohesion_normalized


def roulette_wheel_selection(population, similarity_matrix, summary_size):
    fitness_sum = sum([fitness(x, similarity_matrix, summary_size) for x in population])
    selection = random.choices(population, weights=[fitness(x, similarity_matrix, summary_size) / fitness_sum for x in population], k=2)
    return selection[0], selection[1]


def bits_in_individual(individual):
    return individual.count(True)


def one_point_crossover(parent1, parent2, summary_size):
    good_number_of_bits = False
    while not good_number_of_bits:
        point = random.randrange(0, len(parent1))

        child1 = copy.copy(parent1[0:point])
        child1.extend(copy.copy(parent2[point:len(parent1)]))

        child2 = copy.copy(parent2[0:point])
        child2.extend(copy.copy(parent1[point:len(parent2)]))

        good_number_of_bits = bits_in_individual(child1) == summary_size

    return child1, child2


def mutate(individual):
    point = random.randrange(0, len(individual))
    neighbor = random.sample([1, -1], 1)[0]
    while not (0 <= point + neighbor < len(individual)):
        point = random.randrange(0, len(individual))
        neighbor = random.sample([1, -1], 1)[0]
    individual[point], individual[point + neighbor] = individual[point + neighbor], individual[point]
    return individual


def iteration(population, similarity_matrix, summary_size):
    population.sort(key=lambda individual: fitness(individual, similarity_matrix, summary_size), reverse=True)
    best_two = population[0], population[1]
    del population[:2]  # remove the elites since we always keep them
    parent1, parent2 = roulette_wheel_selection(population, similarity_matrix, summary_size)
    child1, child2 = one_point_crossover(parent1, parent2, summary_size)
    mutate(child1)
    mutate(child2)
    del population[-2:]  # remove the most unfit 2 individuals
    population.append(best_two[0])
    population.append(best_two[1])
    population.append(child1)
    population.append(child2)


def summary_from_individual(best_individual, text_as_sentences):
    return preprocessing.concatenate_text_as_array([text_as_sentences[i] for i in range(len(best_individual)) if best_individual[i] is True])


def generate_population(number_of_sentences, summary_size, population_size=100):
    return [generate_individual(number_of_sentences, summary_size) for _ in range(population_size)]


def generate_summary_evolutionary(sentences_as_embeddings, text_as_sentences_without_footnotes, summary_size, number_of_iterations=100):
    start_time = time.time()
    similarity_matrix = generate_similarity_matrix_for_evolutionary_algorithm(sentences_as_embeddings)
    population = generate_population(len(similarity_matrix), summary_size)
    for i in range(number_of_iterations):
        iteration(population, similarity_matrix, summary_size)
    best_individual = max(population, key=lambda individual: fitness(individual, similarity_matrix, summary_size))
    generated_summary = summary_from_individual(best_individual, text_as_sentences_without_footnotes)
    print("Evolutionary algorithm took ", time.time() - start_time, "s")
    return generated_summary
