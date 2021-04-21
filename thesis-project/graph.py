import numpy as np
import oslom
from scipy import spatial
import time
import infomap as im
import networkx
import processing
import community as community_louvain
from argparse import Namespace
from cdlib import algorithms


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


def summary_from_indexes(sentence_indexes_sorted_by_score, text_as_sentences_without_footnotes):
    return processing.concatenate_text_as_array([text_as_sentences_without_footnotes[i] for i in sentence_indexes_sorted_by_score])


def get_clusters_with_max_coefficients(clusters, clustering_coefficients):
    result = {}
    for cluster, nodes in clusters.items():
        if len(nodes) > 0:
            maxi = 0
            for node in nodes:
                if clustering_coefficients[node] > maxi and node < len(clustering_coefficients):
                    maxi = clustering_coefficients[node]
            result[cluster] = maxi
    return result


def best_from_cluster(clustering_coefficients_for_each_node, clusters, cluster_number):
    nodes = {k: v for k, v in clustering_coefficients_for_each_node.items() if k in clusters[cluster_number]}
    res = list({k: v for k, v in sorted(nodes.items(), key=lambda item: item[1], reverse=True)}.keys())
    if len(res) > 0:
        return res[0]


def remove_best(best, cluster_number, clusters, clustering_coefficients_for_each_node):
    clusters[cluster_number] = [item for item in clusters[cluster_number] if item != best]
    clustering_coefficients_for_each_node.pop(best)


def get_clustering_data(similarity_matrix):
    graph = networkx.from_numpy_array(similarity_matrix)
    clustering_coefficients_for_each_node = networkx.clustering(graph)
    average_clustering_coefficient = networkx.average_clustering(graph)
    return clustering_coefficients_for_each_node, average_clustering_coefficient, graph


def get_average_clustering_coefficient(coefficients_of_clusters):
    return np.average(list(coefficients_of_clusters.values()))


def get_clusters(community_graph, strategy):
    if strategy == "aslpaw":
        clusters = algorithms.aslpaw(community_graph)
    elif strategy == "label_propagation":
        clusters = algorithms.label_propagation(community_graph)
    elif strategy == "greedy_modularity":
        clusters = algorithms.greedy_modularity(community_graph)
    elif strategy == "markov_clustering":
        clusters = algorithms.markov_clustering(community_graph)
    elif strategy == "walktrap":
        clusters = algorithms.walktrap(community_graph)
    elif strategy == "leiden":
        clusters = algorithms.leiden(community_graph)
    else:
        clusters = algorithms.infomap(community_graph)
    clusters = clusters.communities
    result = {}
    for i in range(len(clusters)):
        result[i] = clusters[i]
    return result


def generate_summary_graph(sentences_as_embeddings, text_as_sentences_without_footnotes, summary_size, cluster_strategy="infomap", threshold=0.3):
    print(cluster_strategy)
    start_time = time.time()
    similarity_matrix = generate_similarity_matrix_for_graph_algorithm(sentences_as_embeddings, threshold)
    clustering_coefficients_for_each_node, _, community_graph = get_clustering_data(similarity_matrix)
    clusters = get_clusters(community_graph, cluster_strategy)
    solution = []
    while True:
        coefficients_of_clusters = get_coefficients_for_clusters_sorted(clustering_coefficients_for_each_node, clusters)
        average_clustering_coefficient = get_average_clustering_coefficient(coefficients_of_clusters)
        for cluster_number in list(coefficients_of_clusters):
            if len(solution) < summary_size:

                if coefficients_of_clusters[cluster_number] < average_clustering_coefficient:
                    del coefficients_of_clusters[cluster_number]
                    clusters.pop(cluster_number)
                elif len(clusters[cluster_number]) > 0:
                    best = best_from_cluster(clustering_coefficients_for_each_node, clusters, cluster_number)
                    solution.append(best)
                    remove_best(best, cluster_number, clusters, clustering_coefficients_for_each_node)
        if len(solution) >= summary_size:
            break
    solution.sort()
    summary = summary_from_indexes(solution, text_as_sentences_without_footnotes)
    print("Graphs algorithm took ", time.time() - start_time, "s")
    return summary


def get_coefficients_for_clusters_sorted(clustering_coefficients_for_each_node, clusters):
    coefficients_of_clusters = get_clusters_with_max_coefficients(clusters, clustering_coefficients_for_each_node)
    coefficients_of_clusters = {k: v for k, v in
                                sorted(coefficients_of_clusters.items(), key=lambda item: item[1], reverse=True)}
    return coefficients_of_clusters


# def generate_summary_graph_text_rank(sentences_as_embeddings, text_as_sentences_without_footnotes, summary_size):
#     start_time = time.time()
#     similarity_matrix = generate_similarity_matrix_for_graph_algorithm(sentences_as_embeddings, 0)
#     graph = networkx.from_numpy_array(similarity_matrix)
#     scores = networkx.pagerank(graph)
#     sentence_indexes_sorted_by_score = indexes_from_pagerank(scores, summary_size)
#     summary = summary_from_indexes(sentence_indexes_sorted_by_score, text_as_sentences_without_footnotes)
#     print("Text rank algorithm took ", time.time() - start_time, "s")
#     return summary
