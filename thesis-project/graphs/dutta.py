import time
import graphs.graph_utils as graph_utils
import numpy as np
import processing.processing_utils as processing
from nltk.tag import pos_tag


def generate_summary_dutta(text_as_sentences_without_footnotes, summary_size, threshold=0.3):
    start_time = time.time()
    sentences = sentences_for_dutta(text_as_sentences_without_footnotes)
    embeddings = get_embeddings(sentences)
    similarity_matrix = generate_similarity_matrix_for_dutta(embeddings, threshold)
    clustering_coefficients_for_each_node, _, community_graph = graph_utils.get_clustering_data(similarity_matrix)
    clusters = graph_utils.get_clusters(community_graph, "infomap")
    solution = []
    while True:
        coefficients_of_clusters = graph_utils.get_coefficients_for_clusters_sorted(clustering_coefficients_for_each_node, clusters)
        average_clustering_coefficient = graph_utils.get_average_clustering_coefficient(coefficients_of_clusters)
        for cluster_number in list(coefficients_of_clusters):
            if len(solution) < summary_size:

                if coefficients_of_clusters[cluster_number] < average_clustering_coefficient:
                    del coefficients_of_clusters[cluster_number]
                    clusters.pop(cluster_number)
                elif len(clusters[cluster_number]) > 0:
                    best = graph_utils.best_from_cluster(clustering_coefficients_for_each_node, clusters, cluster_number)
                    solution.append(best)
                    graph_utils.remove_best(best, cluster_number, clusters, clustering_coefficients_for_each_node)
        if len(solution) >= summary_size or len(solution) >= len(
                text_as_sentences_without_footnotes) or graph_utils.no_available_nodes(clusters):
            break
    solution.sort()
    summary = graph_utils.summary_from_indexes(solution, text_as_sentences_without_footnotes)
    print("Graphs algorithm took ", time.time() - start_time, "s")
    return summary


def generate_similarity_matrix_for_dutta(sentences_as_embeddings, threshold):
    number_of_sentences = len(sentences_as_embeddings)
    similarity_matrix = np.zeros([number_of_sentences, number_of_sentences])
    for i, row_embedding in enumerate(sentences_as_embeddings):
        for j, column_embedding in enumerate(sentences_as_embeddings):
            if i == j:
                similarity_matrix[i][j] = 0
            else:
                value = cosine_similarity(row_embedding, column_embedding)
                similarity_matrix[i][j] = value if value > threshold else 0
    return similarity_matrix


def get_words_set(sentences):
    words = []
    for sentence in sentences:
        for word in sentence:
            if word not in words:
                words.append(word)
    return words


def term_position(term, word_set):
    for i in range(len(word_set)):
        if word_set[i] == term:
            return i
    return None


def get_embeddings(sentences):
    words_set = get_words_set(sentences)
    embeddings = []
    for sentence in sentences:
        embedding = [0 for _ in range(len(words_set))]
        for word in sentence:
            embedding[term_position(word, words_set)] = 1
        embeddings.append(embedding)
    return embeddings


def sentences_for_dutta(text_as_sentences_without_footnotes):
    result = []
    for sentence in text_as_sentences_without_footnotes:
        sentence = processing.remove_punctuation(sentence)
        sentence = processing.split_in_tokens(sentence)
        sentence = processing.remove_stop_words(sentence)
        tagged_sentence = pos_tag(sentence)
        sentence_without_proper_nouns = [word for word, pos in tagged_sentence if pos != 'NNP']
        result.append(sentence_without_proper_nouns)
    return result


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