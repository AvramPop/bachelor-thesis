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


# embed = hub.load("/home/dani/Desktop/code/scoala/licenta/use")


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
    # sentences = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[.?!])\s', text)  # this regex can be improved
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


# def sentence_to_embedding(sentence):
#     return embed([sentence]).numpy()[0]


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


def generate_individual(text_size, summary_size):
    individual = np.zeros(text_size, dtype=bool)
    bits = random.sample(range(0, text_size), summary_size)
    for i in bits:
        individual[i] = True
    return individual


def fitness(individual):
    return 1


def roulette_wheel_selection(population):
    pass


def bits_in_individual(individual):
    return individual.count(True)


def one_point_crossover(parent1, parent2, bits):
    good_number_of_bits = False
    while not good_number_of_bits:
        point = random.randint(0, len(parent1))
        print(point)

        child1 = copy.copy(parent1[0:point])
        child1.extend(copy.copy(parent2[point:len(parent1)]))

        child2 = copy.copy(parent2[0:point])
        child2.extend(copy.copy(parent1[point:len(parent2)]))

        good_number_of_bits = bits_in_individual(child1) == bits

    return child1, child2


def mutate(individual):
    point = random.randint(0, len(individual))
    neighbor = random.sample([1, -1], 1)[0]
    while not (0 <= point + neighbor < len(individual)):
        point = random.randint(0, len(individual))
        neighbor = random.sample([1, -1], 1)
    individual[point], individual[point + neighbor] = individual[point + neighbor], individual[point]
    return individual


def main():
    summary_size = 4
    lines = read_file_line_by_line("/home/dani/Desktop/code/scoala/licenta/bachelor-thesis/thesis-project/resources/article-english-with-greek.in")
    text = concatenate_text_as_array(lines)
    text = remove_footnotes(text)
    text_as_sentences = parse_text_to_sentences(text, 1)
    text_as_sentences_without_footnotes = list(text_as_sentences)
    sentences_as_embeddings = []
    for sentence in text_as_sentences:
        print(sentence)
        sentence = remove_punctuation(sentence)
        sentence = split_in_tokens(sentence)
        sentence = tokens_to_lower_case(sentence)
        sentence = remove_stop_words(sentence)
        sentence = transliterate_non_english_words(sentence)
        # sentence = lemmatize(sentence)
        sentence = concatenate_text_as_array(sentence)
        # sentence = sentence_to_embedding(sentence)
        sentences_as_embeddings.append(sentence)

    number_of_sentences = len(sentences_as_embeddings)
    similarity_matrix = generate_similarity_matrix(sentences_as_embeddings)

    number_of_iterations = 10
    population = [generate_individual(number_of_sentences, summary_size) for _ in range(20)]
    for _ in range(number_of_iterations):
        population.sort(key=lambda individual: fitness(individual), reverse=True)
        best_two = (population[0], population[1])
        two_parents = roulette_wheel_selection(population[2:])




# main()
print(mutate([False, True, True, False, False]))