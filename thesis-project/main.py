import re
import string
import stanza
import nltk.data
import unidecode
import tensorflow_hub as hub
import numpy as np
from scipy import spatial
import random
import copy
from pythonrouge.pythonrouge import Pythonrouge


embed = hub.load("/home/dani/Desktop/code/scoala/licenta/use")
summary_size = 6


# read a file to an array in which each item is a line from that respective file
def read_file_line_by_line(filename):
    content = []
    with open(filename) as f:
        for line in f:
            content.append(line.strip())
    return content


# join all elements of an array, separated by a space
def concatenate_text_as_array(text):
    return ' '.join(text)


# parse a string to an array in which each element is a group of sentences_in_batch sentences
def parse_text_to_sentences(text, sentences_in_batch):
    result = []
    sentences = nltk.sent_tokenize(text)

    for i in range(0, len(sentences), sentences_in_batch):
        temp = ''
        for j in range(sentences_in_batch):
            if i + j < len(sentences):
                temp += sentences[i + j] + " "
        result.append(temp)
    return result


# remove all punctuation from a text
def remove_punctuation(text):
    text.replace("'", "")
    text = text.translate(str.maketrans('', '', string.punctuation + r"—"))
    return text


# split a given text to an array in which each element is a word
def split_in_tokens(text):
    text = text.split()
    return text


# remove all stop words from given list of words, using as reference NLTK data
def remove_stop_words(tokens):
    stop_words = read_file_line_by_line("/home/dani/Desktop/code/scoala/licenta/bachelor-thesis/thesis-project/resources/util/stop-words.txt")
    return [word for word in tokens if word not in stop_words]


# lowercase all the words given
def tokens_to_lower_case(tokens):
    return [word.casefold() for word in tokens]


def is_english(s):
    return s.isascii()


# lemmatize given list of words using Stanza (Standford NLP)
def lemmatize(words):
    nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma', tokenize_pretokenized=True)
    doc = nlp([words])
    return [word.lemma for sent in doc.sentences for word in sent.words]


def transliterate_non_english_words(relevant_tokens):
    for i in range(len(relevant_tokens)):
        if not is_english(relevant_tokens[i]):
            relevant_tokens[i] = unidecode.unidecode(relevant_tokens[i])
    return relevant_tokens


def sentence_to_embedding(sentence):
    return embed([sentence]).numpy()[0]


def remove_footnotes(text):
    return re.sub(r"([a-zA-Z?!;,.\"])[0-9]*", r"\1", text)


def generate_similarity_matrix(sentences_as_embeddings):
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


def generate_individual(text_size):
    individual = np.zeros(text_size, dtype=bool)
    bits = random.sample(range(0, text_size), summary_size)
    for i in bits:
        individual[i] = True
    return individual.tolist()


def fitness(individual, similarity_matrix, a=0.34, b=0.33, c=0.33):
    return (a * cohesion(individual, similarity_matrix) + b * readability(individual, similarity_matrix) + c * sentence_position(individual)) / (a + b + c)


def max_weight_dag_util(start, weights, similarity_matrix):
    for i in range(len(similarity_matrix)):
        if similarity_matrix[start][i] > 0 and weights[i] == 0:
            weights[i] = max(weights[i], weights[start] + similarity_matrix[start][i])
            max_weight_dag_util(i, weights, similarity_matrix)


def max_weight_dag(similarity_matrix):
    weights = [0 for _ in range(len(similarity_matrix))]
    max_weight_dag_util(0, weights, similarity_matrix)
    return max(weights)


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
    return result


def cohesion(individual, similarity_matrix):
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


def roulette_wheel_selection(population, similarity_matrix):
    fitness_sum = sum([fitness(x, similarity_matrix) for x in population])
    selection = random.choices(population, weights=[fitness(x, similarity_matrix) / fitness_sum for x in population], k=2)
    return selection[0], selection[1]


def bits_in_individual(individual):
    return individual.count(True)


def one_point_crossover(parent1, parent2):
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


def iteration(population, similarity_matrix):
    population.sort(key=lambda individual: fitness(individual, similarity_matrix), reverse=True)
    best_two = population[0], population[1]
    del population[:2] # remove the elites since we always keep them
    parent1, parent2 = roulette_wheel_selection(population, similarity_matrix)
    child1, child2 = one_point_crossover(parent1, parent2)
    mutate(child1)
    mutate(child2)
    del population[-2:]  # remove the most unfit 2 individuals
    population.append(best_two[0])
    population.append(best_two[1])
    population.append(child1)
    population.append(child2)


def summary_from_individual(best_individual, text_as_sentences):
    return concatenate_text_as_array([text_as_sentences[i] for i in range(len(best_individual)) if best_individual[i] is True])


def generate_population(number_of_sentences, population_size=20):
    return [generate_individual(number_of_sentences) for _ in range(population_size)]


def read_file_to_text(filename):
    title_lines = read_file_line_by_line(filename)
    title = concatenate_text_as_array(title_lines)
    title = title.casefold()
    return title


def main():
    title = read_file_to_text("/home/dani/Desktop/code/scoala/licenta/bachelor-thesis/thesis-project/resources/articles/1-c.txt")
    abstract = read_file_to_text("/home/dani/Desktop/code/scoala/licenta/bachelor-thesis/thesis-project/resources/articles/1-b.txt")
    text_lines = read_file_line_by_line("/home/dani/Desktop/code/scoala/licenta/bachelor-thesis/thesis-project/resources/articles/1-a.txt")
    text = concatenate_text_as_array(text_lines)
    text = remove_footnotes(text)
    text_as_sentences = parse_text_to_sentences(text, 1)
    text_as_sentences_without_footnotes = list(text_as_sentences)
    sentences_as_embeddings = []
    for sentence in text_as_sentences:
        sentence = remove_punctuation(sentence)
        sentence = split_in_tokens(sentence)
        sentence = tokens_to_lower_case(sentence)
        sentence = remove_stop_words(sentence)
        sentence = transliterate_non_english_words(sentence)
        sentence = lemmatize(sentence)
        sentence = concatenate_text_as_array(sentence)
        sentence = sentence_to_embedding(sentence)
        sentences_as_embeddings.append(sentence)

    number_of_sentences = len(sentences_as_embeddings)
    similarity_matrix = generate_similarity_matrix(sentences_as_embeddings)

    number_of_iterations = 10
    population = generate_population(number_of_sentences)
    for i in range(number_of_iterations):
        print("iteration " + str(i))
        iteration(population, similarity_matrix)
    best_individual = max(population, key=lambda individual: fitness(individual, similarity_matrix))
    generated_summary = summary_from_individual(best_individual, text_as_sentences_without_footnotes)
    print(abstract)
    print(generated_summary)
    rouge = Pythonrouge(summary_file_exist=False,
                        summary=[[generated_summary]], reference=[[[abstract]]])
    score = rouge.calc_score()
    print(score)


main()
