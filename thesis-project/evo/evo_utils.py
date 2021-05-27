import random
import copy
import numpy as np
from scipy import spatial
import processing.processing_utils as processing


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


def mutate(individual, mutate_threshold=0.6):
    r = random.uniform(0, 1)
    if r > mutate_threshold:
        point = random.randrange(0, len(individual))
        neighbor = random.sample([1, -1], 1)[0]
        while not (0 <= point + neighbor < len(individual)):
            point = random.randrange(0, len(individual))
            neighbor = random.sample([1, -1], 1)[0]
        individual[point], individual[point + neighbor] = individual[point + neighbor], individual[point]
    return individual


def bits_in_individual(individual):
    return individual.count(True)


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


def generate_individual(text_size, summary_size):
    individual = np.zeros(text_size, dtype=bool)
    bits = random.sample(range(0, text_size), summary_size)
    for i in bits:
        individual[i] = True
    return individual.tolist()


def title_relation(individual, sentences_as_embeddings, title_embedding):
    indices = [i for i in range(len(individual)) if individual[i]]
    title_relation_metric = 0
    for i in indices:
        title_relation_metric += cosine_distance(sentences_as_embeddings[i], title_embedding)
    title_relation_metric = title_relation_metric / len(indices)
    backup_text_embeddings = copy.copy(sentences_as_embeddings)
    backup_text_embeddings.sort(key=lambda x: cosine_distance(x, title_embedding), reverse=True)
    backup_text_embeddings = backup_text_embeddings[:len(indices)]
    best_title_relation_score = 0
    for sentence in backup_text_embeddings:
        best_title_relation_score += cosine_distance(sentence, title_embedding)
    best_title_relation_score = best_title_relation_score / len(indices)
    return title_relation_metric / best_title_relation_score


def sentence_length_metric(individual, text_as_sentences):
    indices = [i for i in range(len(individual)) if individual[i]]
    sentences = [text_as_sentences[i] for i in indices]
    lengths = [len(sentence.split()) for sentence in sentences]
    average = np.average(lengths)
    std = np.std(lengths)
    length_metric = 0
    for length in lengths:
        length_metric += (1 - np.exp((-length - average) / std)) / (1 + np.exp((-length - average) / std))

    backup_text = copy.copy(text_as_sentences)
    backup_text.sort(key=lambda x: len(x.split()), reverse=True)
    backup_text = backup_text[:len(indices)]
    lengths_best = [len(sentence.split()) for sentence in backup_text]
    average_best = np.average(lengths_best)
    std_best = np.std(lengths_best)
    length_metric_best = 0
    for length in lengths_best:
        length_metric_best += (1 - np.exp((-length - average_best) / std_best)) / (1 + np.exp((-length - average_best) / std_best))

    return length_metric / length_metric_best


def generate_population(number_of_sentences, summary_size, population_size):
    return [generate_individual(number_of_sentences, summary_size) for _ in range(population_size)]


def cosine_distance(sentence1, sentence2):
    return 1 - spatial.distance.cosine(sentence2, sentence1)


def summary_from_individual(best_individual, text_as_sentences):
    return processing.concatenate_text_as_array(
        [text_as_sentences[i] for i in range(len(best_individual)) if best_individual[i] is True])