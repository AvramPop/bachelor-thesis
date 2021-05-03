import random
import copy
import numpy as np
from scipy import spatial
import time
import processing
import evolutionary


def roulette_wheel_selection(population, similarity_matrix, summary_size, a, b):
    fitness_sum = sum([fitness(x, similarity_matrix, summary_size, a, b) for x in population])
    selection = random.choices(population,
                               weights=[fitness(x, similarity_matrix, summary_size, a, b) / fitness_sum for x in population])
    return selection[0]


def iteration(population, similarity_matrix, summary_size, text_as_sentences_without_footnotes, a, b):
    population.sort(key=lambda individual: fitness(individual, similarity_matrix, summary_size, a, b),
                    reverse=True)
    best_two = population[0], population[1]
    del population[:2]  # remove the elites since we always keep them
    parent1 = roulette_wheel_selection(population, similarity_matrix, summary_size, a, b)
    parent2 = roulette_wheel_selection(population, similarity_matrix, summary_size, a, b)
    child1, child2 = evolutionary.one_point_crossover(parent1, parent2, summary_size)
    evolutionary.mutate(child1)
    evolutionary.mutate(child2)
    del population[-2:]  # remove the most unfit 2 individuals
    population.append(best_two[0])
    population.append(best_two[1])
    population.append(child1)
    population.append(child2)


def fitness(individual, similarity_matrix, summary_size, a, b):
    try:
        cohesion_value = evolutionary.cohesion(individual, similarity_matrix, summary_size)
    except:
        cohesion_value = 0
        a = 0

    try:
        readability_value = evolutionary.readability(individual, similarity_matrix)
    except:
        readability_value = 0
        b = 0

    return a * cohesion_value + b * readability_value


def generate_summary_evolutionary(text_as_sentences_without_footnotes, summary_size, number_of_iterations=25,
                                             population_size=20, a=0.05, b=0.5):
    start_time = time.time()
    similarity_matrix = generate_similarity_matrix(text_as_sentences_without_footnotes)
    population = evolutionary.generate_population(len(similarity_matrix), summary_size, population_size)
    for i in range(number_of_iterations):
        iteration(population, similarity_matrix, summary_size, text_as_sentences_without_footnotes, a, b)
    best_individual = max(population,
                          key=lambda individual: fitness(individual, similarity_matrix, summary_size, a, b))
    generated_summary = summary_from_individual(best_individual, text_as_sentences_without_footnotes)
    print("Chatterjee algorithm took ", time.time() - start_time, "s")
    return generated_summary


def summary_from_individual(best_individual, text_as_sentences):
    return processing.concatenate_text_as_array(
        [text_as_sentences[i] for i in range(len(best_individual)) if best_individual[i] is True])


def cosine_similarity(i_sentence, j_sentence):
    term1 = 0
    for i in range(len(i_sentence)):
        term1 += i_sentence[i] * j_sentence[i]
    term2_i = 0
    term2_j = 0
    for i in range(len(i_sentence)):
        term2_i += i_sentence[i] * i_sentence[i]
        term2_j += j_sentence[i] * j_sentence[i]
    term2_i = np.sqrt(term2_i)
    term2_j = np.sqrt(term2_j)
    term2 = term2_i * term2_j
    return term1 / term2


def get_words_set(sentences):
    words = []
    for sentence in sentences:
        sentence = processing.remove_punctuation(sentence)
        sentence = processing.split_in_tokens(sentence)
        sentence = processing.remove_stop_words(sentence)
        for word in sentence:
            if word not in words:
                words.append(word)
    return words


def term_frequency(term, sentence):
    sentence = processing.remove_punctuation(sentence)
    sentence = processing.split_in_tokens(sentence)
    term_count = sentence.count(term)
    maxi = 1
    for word in sentence:
        temp_count = sentence.count(word)
        if temp_count > maxi:
            maxi = temp_count
    return term_count / maxi


def get_sentence_tokens(sentence):
    sentence = processing.remove_punctuation(sentence)
    return processing.split_in_tokens(sentence)


def inverse_sentence_frequency(term, sentences):
    n = 0
    for sentence in sentences:
        sentence = get_sentence_tokens(sentence)
        if term in sentence:
            n += 1
    N = len(sentences)
    if n == 0:
        n = 1
    return np.log(N / n)


def get_index_terms(sentence):
    sentence = processing.remove_punctuation(sentence)
    sentence = processing.split_in_tokens(sentence)
    return processing.remove_stop_words(sentence)


def term_position(term, word_set):
    for i in range(len(word_set)):
        if word_set[i] == term:
            return i
    return None


def get_weight(term, sentence, sentences):
    return term_frequency(term, sentence) * inverse_sentence_frequency(term, sentences)


def get_embeddings(sentences):
    embeddings = []
    word_set = get_words_set(sentences)
    embedding_size = len(word_set)
    for sentence in sentences:
        sentence_embedding = [0 for _ in range(embedding_size)]
        index_terms = get_index_terms(sentence)
        for term in index_terms:
            sentence_embedding[term_position(term, word_set)] = get_weight(term, sentence, sentences)
        embeddings.append(sentence_embedding)
    return embeddings


def generate_similarity_matrix(sentences):
    number_of_sentences = len(sentences)
    embeddings = get_embeddings(sentences)
    similarity_matrix = np.zeros([number_of_sentences, number_of_sentences])
    for i, i_sentence in enumerate(embeddings):
        for j, j_sentence in enumerate(embeddings):
            similarity_matrix[i][j] = cosine_similarity(i_sentence, j_sentence)
    for i in range(number_of_sentences):
        for j in range(number_of_sentences):
            if j <= i:
                similarity_matrix[i][j] = 0
    return similarity_matrix
