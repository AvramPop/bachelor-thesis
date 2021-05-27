from cdlib import algorithms
import networkx
import numpy as np
import processing.processing_utils as processing


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


def get_clustering_data(similarity_matrix):
    graph = networkx.from_numpy_array(similarity_matrix)
    clustering_coefficients_for_each_node = networkx.clustering(graph)
    average_clustering_coefficient = networkx.average_clustering(graph)
    return clustering_coefficients_for_each_node, average_clustering_coefficient, graph


def get_coefficients_for_clusters_sorted(clustering_coefficients_for_each_node, clusters):
    coefficients_of_clusters = get_clusters_with_max_coefficients(clusters, clustering_coefficients_for_each_node)
    coefficients_of_clusters = {k: v for k, v in
                                sorted(coefficients_of_clusters.items(), key=lambda item: item[1], reverse=True)}
    return coefficients_of_clusters


def get_average_clustering_coefficient(coefficients_of_clusters):
    return np.average(list(coefficients_of_clusters.values()))


def best_from_cluster(clustering_coefficients_for_each_node, clusters, cluster_number):
    nodes = {k: v for k, v in clustering_coefficients_for_each_node.items() if k in clusters[cluster_number]}
    res = list({k: v for k, v in sorted(nodes.items(), key=lambda item: item[1], reverse=True)}.keys())
    if len(res) > 0:
        return res[0]


def remove_best(best, cluster_number, clusters, clustering_coefficients_for_each_node):
    clusters[cluster_number] = [item for item in clusters[cluster_number] if item != best]
    clustering_coefficients_for_each_node.pop(best)


def no_available_nodes(clusters):
    for k, v in clusters.items():
        if len(v) > 0:
            return False
    return True


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
