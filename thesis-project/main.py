import evolutionary
import graph
import processing
import time
import ui
import duc
import chatterjee


# def main():
#     number_of_texts = 20
#     evolutionary_scores = []
#     graph_scores = []
#     start_time = time.time()
#     for i in range(1, number_of_texts + 1):
#         print("Current article: " + str(i))
#         sentences_as_embeddings, text_as_sentences_without_footnotes, abstract, title, title_embedding, rough_abstract = processing.prepare_data(i)
#         print("text length is: " + str(len(text_as_sentences_without_footnotes)))
#
#         generated_summary_evolutionary = evolutionary.generate_summary_evolutionary(sentences_as_embeddings, title_embedding, text_as_sentences_without_footnotes, processing.number_of_sentences_in_text(abstract))
#         generated_summary_graph = graph.generate_summary_graph(sentences_as_embeddings, text_as_sentences_without_footnotes, processing.number_of_sentences_in_text(abstract))
#
#         score_evolutionary = processing.rouge_score(generated_summary_evolutionary, abstract)
#         score_graphs = processing.rouge_score(generated_summary_graph, abstract)
#
#         print(score_evolutionary)
#         print(score_graphs)
#
#         print(generated_summary_evolutionary)
#         print(generated_summary_graph)
#
#         evolutionary_scores.append(score_evolutionary)
#         graph_scores.append(score_graphs)
#
#     print("RESULTS:")
#
#     print("Evolutionary average score:")
#     print(processing.final_results(evolutionary_scores))
#     print("Graphs average score: ")
#     print(processing.final_results(graph_scores))
#
#     print(str(number_of_texts), " articles processing took exactly ", time.time() - start_time, "s")
#
#
# main()

# def main_graphs():
#     number_of_texts = 20
#     # evolutionary_scores = []
#     graph_scores1 = []
#     graph_scores2 = []
#     graph_scores3 = []
#     graph_scores4 = []
#     graph_scores5 = []
#     graph_scores6 = []
#     graph_scores7 = []
#
#     start_time = time.time()
#     for i in range(1, number_of_texts + 1):
#         print("Current article: " + str(i))
#         sentences_as_embeddings, text_as_sentences_without_footnotes, abstract, title, title_embedding, rough_abstract = processing.prepare_data(i)
#         print("text length is: " + str(len(text_as_sentences_without_footnotes)))
#
#         try:
#             generated_summary_graph1 = graph.generate_summary_graph(sentences_as_embeddings, text_as_sentences_without_footnotes, processing.number_of_sentences_in_text(abstract), cluster_strategy="aslpaw")
#             generated_summary_graph2 = graph.generate_summary_graph(sentences_as_embeddings, text_as_sentences_without_footnotes, processing.number_of_sentences_in_text(abstract), cluster_strategy="label_propagation")
#             generated_summary_graph3 = graph.generate_summary_graph(sentences_as_embeddings, text_as_sentences_without_footnotes, processing.number_of_sentences_in_text(abstract), cluster_strategy="greedy_modularity")
#             generated_summary_graph4 = graph.generate_summary_graph(sentences_as_embeddings, text_as_sentences_without_footnotes, processing.number_of_sentences_in_text(abstract), cluster_strategy="markov_clustering")
#             generated_summary_graph5 = graph.generate_summary_graph(sentences_as_embeddings, text_as_sentences_without_footnotes, processing.number_of_sentences_in_text(abstract), cluster_strategy="walktrap")
#             generated_summary_graph6 = graph.generate_summary_graph(sentences_as_embeddings, text_as_sentences_without_footnotes, processing.number_of_sentences_in_text(abstract), cluster_strategy="leiden")
#             generated_summary_graph7 = graph.generate_summary_graph(sentences_as_embeddings, text_as_sentences_without_footnotes, processing.number_of_sentences_in_text(abstract), cluster_strategy="infomap")
#
#
#             score_graphs1 = processing.rouge_score(generated_summary_graph1, abstract)
#             score_graphs2 = processing.rouge_score(generated_summary_graph2, abstract)
#             score_graphs3 = processing.rouge_score(generated_summary_graph3, abstract)
#             score_graphs4 = processing.rouge_score(generated_summary_graph4, abstract)
#             score_graphs5 = processing.rouge_score(generated_summary_graph5, abstract)
#             score_graphs6 = processing.rouge_score(generated_summary_graph6, abstract)
#             score_graphs7 = processing.rouge_score(generated_summary_graph7, abstract)
#
#
#             # print(score_graphs)
#
#             graph_scores1.append(score_graphs1)
#             graph_scores2.append(score_graphs2)
#             graph_scores3.append(score_graphs3)
#             graph_scores4.append(score_graphs4)
#             graph_scores5.append(score_graphs5)
#             graph_scores6.append(score_graphs6)
#             graph_scores7.append(score_graphs7)
#         except:
#             print("exception at i = " + str(i) + ". skipping.")
#
#
#     print("RESULTS:")
#
#     # print("Graphs average score: ")
#     print(processing.final_results(graph_scores1))
#     print(processing.final_results(graph_scores2))
#     print(processing.final_results(graph_scores3))
#     print(processing.final_results(graph_scores4))
#     print(processing.final_results(graph_scores5))
#     print(processing.final_results(graph_scores6))
#     print(processing.final_results(graph_scores7))
#
#     print(str(number_of_texts), " articles processing took exactly ", time.time() - start_time, "s")
#
# main_graphs()

# def main_ui():
#     titles = processing.get_titles(20)
#     ui.launch_ui(titles)
#
#
# main_ui()


# def main_duc():
#     docs, summaries = duc.get_duc_data()
#     number_of_texts = len(docs)
#     evolutionary_scores = []
#     graph_scores = []
#     start_time = time.time()
#     for i in range(number_of_texts):
#         sentences_as_embeddings, text_as_sentences_without_footnotes, abstract, title, title_embedding, rough_abstract = processing.preprocess_duc(docs[i], duc.get_summary(docs[i], summaries))
#         if sentences_as_embeddings is not None:
#             generated_summary_evolutionary = evolutionary.generate_summary_evolutionary(sentences_as_embeddings, title_embedding, text_as_sentences_without_footnotes, processing.number_of_sentences_in_text(abstract))
#             generated_summary_graph = graph.generate_summary_graph(sentences_as_embeddings, text_as_sentences_without_footnotes, processing.number_of_sentences_in_text(abstract))
#
#             score_evolutionary = processing.rouge_score(generated_summary_evolutionary, abstract)
#             score_graphs = processing.rouge_score(generated_summary_graph, abstract)
#
#             evolutionary_scores.append(score_evolutionary)
#             graph_scores.append(score_graphs)
#             print("Done article: " + str(i))
#
#     print("RESULTS:")
#
#     print("Evolutionary average score:")
#     print(processing.final_results(evolutionary_scores))
#     print("Graphs average score: ")
#     print(processing.final_results(graph_scores))
#
#     print(str(number_of_texts), " articles processing took exactly ", time.time() - start_time, "s")
#
# main_duc()


def main_papers():
    number_of_texts = 5
    scores = []
    start_time = time.time()
    for i in range(1, number_of_texts + 1):
        print("Current article: " + str(i))
        sentences_as_embeddings, text_as_sentences_without_footnotes, abstract, title, title_embedding, rough_abstract = processing.prepare_data(i)
        print("text length is: " + str(len(text_as_sentences_without_footnotes)))

        pagerank = graph.generate_summary_graph_text_rank(text_as_sentences_without_footnotes, processing.number_of_sentences_in_text(abstract))
        if pagerank is None:
            print("FAILURE")
            continue
        else:
            score = processing.rouge_score(pagerank, abstract)

        print(score)

        print(pagerank)

        scores.append(score)

    print("RESULTS:")

    print("Pagerank average score:")
    print(processing.final_results(scores))

    print(str(number_of_texts), " articles processing took exactly ", time.time() - start_time, "s")


main_papers()
