import processing.processing_utils as processing
import networkx
import time
import graphs.graph_utils as graph_utils
import numpy as np


def generate_summary_graph_text_rank(text_as_sentences_without_footnotes, summary_size):
    start_time = time.time()
    similarity_matrix = generate_similarity_matrix_for_pagerank(text_as_sentences_without_footnotes)
    graph = networkx.from_numpy_array(similarity_matrix)
    try:
        scores = networkx.pagerank(graph, max_iter=1000)
    except networkx.exception.PowerIterationFailedConvergence as e:
        return None
    sentence_indexes_sorted_by_score = indexes_from_pagerank(scores, summary_size)
    summary = graph_utils.summary_from_indexes(sentence_indexes_sorted_by_score, text_as_sentences_without_footnotes)
    print("Text rank algorithm took ", time.time() - start_time, "s")
    return summary


# obtain first summary_size sentences by their text rank score
def indexes_from_pagerank(scores, summary_size):
    return sorted([k for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)][:summary_size])


# compute sentence similarity as described in Mihalcea2004
def pagerank_similarity(sentence_i, sentence_j):
    sentence_i = processing.remove_punctuation(sentence_i)
    sentence_i = processing.split_in_tokens(sentence_i)

    sentence_j = processing.remove_punctuation(sentence_j)
    sentence_j = processing.split_in_tokens(sentence_j)

    term1 = 0
    for word in sentence_i:
        if word in sentence_j:
            term1 += 1
    term2 = np.log(len(sentence_i)) + np.log(len(sentence_j))

    return term1 / term2


def generate_similarity_matrix_for_pagerank(text_as_sentences_without_footnotes):
    number_of_sentences = len(text_as_sentences_without_footnotes)
    similarity_matrix = np.zeros([number_of_sentences, number_of_sentences])
    for i, sentence_i in enumerate(text_as_sentences_without_footnotes):
        for j, sentence_j in enumerate(text_as_sentences_without_footnotes):
            if i == j:
                similarity_matrix[i][j] = 0
            else:
                similarity_matrix[i][j] = pagerank_similarity(sentence_i, sentence_j)
    return similarity_matrix
