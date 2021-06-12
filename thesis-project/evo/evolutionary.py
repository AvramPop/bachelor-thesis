import random
import numpy as np
import time
import evo.evo_utils as evo_utils


def generate_similarity_matrix_for_evolutionary_algorithm(sentences_as_embeddings):
    number_of_sentences = len(sentences_as_embeddings)
    similarity_matrix = np.zeros([number_of_sentences, number_of_sentences])
    for i, row_embedding in enumerate(sentences_as_embeddings):
        for j, column_embedding in enumerate(sentences_as_embeddings):
            similarity_matrix[i][j] = evo_utils.cosine_distance(column_embedding, row_embedding)
    for i in range(number_of_sentences):
        for j in range(number_of_sentences):
            if j <= i:
                similarity_matrix[i][j] = 0
    return similarity_matrix


def fitness(individual, similarity_matrix, summary_size, title_embedding, sentences_as_embeddings, text_as_sentences, a=0.05, b=0.35, c=0.2, d=0.35, e=0.05):
    try:
        cohesion_value = evo_utils.cohesion(individual, similarity_matrix, summary_size)
    except:
        cohesion_value = 0
        a = 0

    try:
        readability_value = evo_utils.readability(individual, similarity_matrix)
    except:
        readability_value = 0
        b = 0

    try:
        sentence_position_value = evo_utils.sentence_position(individual)
    except:
        sentence_position_value = 0
        c = 0

    try:
        title_value = evo_utils.title_relation(individual, sentences_as_embeddings, title_embedding)
    except:
        title_value = 0
        d = 0

    try:
        sentence_length_value = evo_utils.sentence_length_metric(individual, text_as_sentences)
    except:
        sentence_length_value = 0
        e = 0

    return a * cohesion_value + b * readability_value + c * sentence_position_value + d * title_value + e * sentence_length_value


def roulette_wheel_selection(population, similarity_matrix, summary_size, title_embedding, sentences_as_embeddings, text_as_sentences_without_footnotes, a, b, c, d, e):
    fitness_sum = sum([fitness(x, similarity_matrix, summary_size, title_embedding, sentences_as_embeddings, text_as_sentences_without_footnotes, a, b, c, d, e) for x in population])
    selection = random.choices(population,
                               weights=[fitness(x, similarity_matrix, summary_size, title_embedding, sentences_as_embeddings, text_as_sentences_without_footnotes, a, b, c, d, e) / fitness_sum for x in population])
    return selection[0]


def rank_selection(population):
    best = population[0]
    medium_score = 0
    for individual in population[1:]:
        medium_score += abs(evo_utils.cosine_distance(best, individual))
    medium_score = medium_score / (len(population) - 1)
    scores = [(medium_score - 2 * (medium_score - 1) * i / (len(population) - 1)) / len(population) for i in range(len(population))]
    selection = random.choices(population, weights=scores)
    return selection[0]


def iteration(population, similarity_matrix, summary_size, title_embedding, sentences_as_embeddings, sentences_without_footnotes, a, b, c, d, e):
    population.sort(key=lambda individual: fitness(individual, similarity_matrix, summary_size, title_embedding, sentences_as_embeddings, sentences_without_footnotes, a, b, c, d, e), reverse=True)
    best_two = population[0], population[1]
    del population[:2]  # remove the elites since we always keep them
    parent1 = roulette_wheel_selection(population, similarity_matrix, summary_size, title_embedding, sentences_as_embeddings, sentences_without_footnotes, a, b, c, d, e)
    parent2 = rank_selection(population)
    child1, child2 = evo_utils.one_point_crossover(parent1, parent2, summary_size)
    evo_utils.mutate(child1)
    evo_utils.mutate(child2)
    del population[-2:]  # remove the most unfit 2 individuals
    population.append(best_two[0])
    population.append(best_two[1])
    population.append(child1)
    population.append(child2)


def generate_summary_evolutionary(sentences_as_embeddings, title_embedding, text_as_sentences_without_footnotes, summary_size,
                                  number_of_iterations=25, population_size=20, a=0.2, b=0.2, c=0.2, d=0.2, e=0.2):
    start_time = time.time()
    similarity_matrix = generate_similarity_matrix_for_evolutionary_algorithm(sentences_as_embeddings)
    population = evo_utils.generate_population(len(similarity_matrix), summary_size, population_size)
    for i in range(number_of_iterations):
        iteration(population, similarity_matrix, summary_size, title_embedding, sentences_as_embeddings, text_as_sentences_without_footnotes, a, b, c, d, e)
    best_individual = max(population, key=lambda individual: fitness(individual, similarity_matrix, summary_size, title_embedding, sentences_as_embeddings, text_as_sentences_without_footnotes, a, b, c, d, e))
    generated_summary = evo_utils.summary_from_individual(best_individual, text_as_sentences_without_footnotes)
    print("Evolutionary algorithm took ", time.time() - start_time, "s")
    return generated_summary
