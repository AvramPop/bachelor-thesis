import numpy as np
from scipy import spatial
import time
import infomap as im
import networkx
import preprocessing


def generate_similarity_matrix_for_graph_algorithm(sentences_as_embeddings):
    number_of_sentences = len(sentences_as_embeddings)
    similarity_matrix = np.zeros([number_of_sentences, number_of_sentences])
    for i, row_embedding in enumerate(sentences_as_embeddings):
        for j, column_embedding in enumerate(sentences_as_embeddings):
            similarity_matrix[i][j] = 1 - spatial.distance.cosine(row_embedding, column_embedding)
    return similarity_matrix


def get_clusters(similarity_matrix, threshold):
    infomap = im.Infomap("--two-level")
    for i in range(len(similarity_matrix)):
        for j in range(len(similarity_matrix)):
            if similarity_matrix[i][j] > threshold:
                infomap.add_link(i, j, similarity_matrix[i][j])
    infomap.run()
    print(f"Found {infomap.num_top_modules} modules with codelength: {infomap.codelength}")
    print("Result")
    print("\n#node module")
    for node in infomap.tree:
        if node.is_leaf:
            print(node.node_id, node.module_id)
    return infomap


def indexes_from_pagerank(scores, summary_size):
    return sorted([k for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)][:summary_size])


def summary_from_indexes(sentence_indexes_sorted_by_score, text_as_sentences_without_footnotes):
    return preprocessing.concatenate_text_as_array([text_as_sentences_without_footnotes[i] for i in sentence_indexes_sorted_by_score])


def generate_summary_graph(sentences_as_embeddings, text_as_sentences_without_footnotes, summary_size, threshold=0):
    start_time = time.time()
    similarity_matrix = generate_similarity_matrix_for_graph_algorithm(sentences_as_embeddings)
    # get_clusters(similarity_matrix, threshold)
    graph = networkx.from_numpy_array(similarity_matrix)
    scores = networkx.pagerank(graph, max_iter=20000)
    sentence_indexes_sorted_by_score = indexes_from_pagerank(scores, summary_size)
    summary = summary_from_indexes(sentence_indexes_sorted_by_score, text_as_sentences_without_footnotes)
    print("Graphs algorithm took ", time.time() - start_time, "s")
    return summary
