import numpy as np
from scipy import spatial
import graphs.graph_utils as graph_utils
import time


def generate_similarity_matrix_for_graph_algorithm(sentences_as_embeddings, threshold):
    number_of_sentences = len(sentences_as_embeddings)
    similarity_matrix = np.zeros([number_of_sentences, number_of_sentences])
    for i, row_embedding in enumerate(sentences_as_embeddings):
        for j, column_embedding in enumerate(sentences_as_embeddings):
            if i == j:
                similarity_matrix[i][j] = 0
            else:
                value = 1 - spatial.distance.cosine(row_embedding, column_embedding)
                similarity_matrix[i][j] = value if value > threshold else 0
    return similarity_matrix


def generate_summary_graph(sentences_as_embeddings, text_as_sentences_without_footnotes, summary_size, cluster_strategy="infomap", threshold=0.3):
    start_time = time.time()
    similarity_matrix = generate_similarity_matrix_for_graph_algorithm(sentences_as_embeddings, threshold)
    clustering_coefficients_for_each_node, _, community_graph = graph_utils.get_clustering_data(similarity_matrix)
    clusters = graph_utils.get_clusters(community_graph, cluster_strategy)
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
        if len(solution) >= summary_size or len(solution) >= len(text_as_sentences_without_footnotes) or graph_utils.no_available_nodes(clusters):
            break
    solution.sort()
    summary = graph_utils.summary_from_indexes(solution, text_as_sentences_without_footnotes)
    print("Graphs algorithm took ", time.time() - start_time, "s")
    return summary
