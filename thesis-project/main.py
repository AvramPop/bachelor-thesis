from evo import evolutionary, chatterjee
from graphs import graph, dutta, textrank
import processing.processing_utils as processing
import time
import ui.ui_driver as ui
from processing import duc
import sys


def ui_driver():
    titles = processing.get_titles(39)
    ui.launch_ui(titles)


def theology_driver(number_of_texts=39):
    print("Theology benchmark")
    evolutionary_scores = []
    graph_scores = []
    text_rank_scores = []
    dutta_scores = []
    chatterjee_scores = []
    start_time = time.time()
    for i in range(1, number_of_texts + 1):
        print("Current article: " + str(i))
        sentences_as_embeddings, text_as_sentences_without_footnotes, abstract, title, title_embedding, rough_abstract = processing.prepare_data(i)

        summary_length = processing.number_of_sentences_in_text(abstract)
        generated_summary_evolutionary = evolutionary.generate_summary_evolutionary(sentences_as_embeddings, title_embedding, text_as_sentences_without_footnotes,
                                                                                    summary_length)
        generated_summary_graph = graph.generate_summary_graph(sentences_as_embeddings, text_as_sentences_without_footnotes,
                                                               summary_length)
        generated_summary_textrank = textrank.generate_summary_graph_text_rank(text_as_sentences_without_footnotes, summary_length)
        generated_summary_dutta = dutta.generate_summary_dutta(text_as_sentences_without_footnotes, summary_length)
        generated_summary_chatterjee = chatterjee.generate_summary_chatterjee(text_as_sentences_without_footnotes, summary_length)

        score_evolutionary = processing.rouge_score(generated_summary_evolutionary, abstract)
        score_graphs = processing.rouge_score(generated_summary_graph, abstract)
        if generated_summary_textrank is not None:
            score_textrank = processing.rouge_score(generated_summary_textrank, abstract)
        else:
            score_textrank = None
        score_dutta = processing.rouge_score(generated_summary_dutta, abstract)
        score_chatterjee = processing.rouge_score(generated_summary_chatterjee, abstract)

        evolutionary_scores.append(score_evolutionary)
        graph_scores.append(score_graphs)
        if score_textrank is not None:
            text_rank_scores.append(score_textrank)
        dutta_scores.append(score_dutta)
        chatterjee_scores.append(score_chatterjee)

    print("RESULTS on the Theology dataset:")

    print("Evolutionary average score:")
    print(processing.final_results(evolutionary_scores))
    print("Graphs average score: ")
    print(processing.final_results(graph_scores))
    print("Textrank average score: [for ONLY " + str(len(text_rank_scores)) + " articles]")
    print(processing.final_results(text_rank_scores))
    print("Dutta average score: ")
    print(processing.final_results(dutta_scores))
    print("Chatterjee average score: ")
    print(processing.final_results(chatterjee_scores))

    print(str(number_of_texts), " articles processing took exactly ", time.time() - start_time, "s")


def duc_driver():
    print("DUC benchmark")
    docs, summaries = duc.get_duc_data()
    number_of_texts = 3  # len(docs)
    evolutionary_scores = []
    graph_scores = []
    text_rank_scores = []
    dutta_scores = []
    chatterjee_scores = []
    start_time = time.time()
    for i in range(number_of_texts):
        print("current article is " + str(i))
        sentences_as_embeddings, text_as_sentences_without_footnotes, abstract, title, title_embedding, rough_abstract = processing.preprocess_duc(docs[i], duc.get_summary(docs[i], summaries))
        if sentences_as_embeddings is not None:
            summary_length = processing.number_of_sentences_in_text(abstract)
            generated_summary_evolutionary = evolutionary.generate_summary_evolutionary(sentences_as_embeddings,
                                                                                        title_embedding,
                                                                                        text_as_sentences_without_footnotes,
                                                                                        summary_length)
            generated_summary_graph = graph.generate_summary_graph(sentences_as_embeddings,
                                                                   text_as_sentences_without_footnotes,
                                                                   summary_length)
            generated_summary_textrank = textrank.generate_summary_graph_text_rank(text_as_sentences_without_footnotes, summary_length)
            generated_summary_dutta = dutta.generate_summary_dutta(text_as_sentences_without_footnotes, summary_length, threshold=0.1)
            generated_summary_chatterjee = chatterjee.generate_summary_chatterjee(text_as_sentences_without_footnotes,
                                                                                  summary_length)

            score_evolutionary = processing.rouge_score(generated_summary_evolutionary, abstract)
            score_graphs = processing.rouge_score(generated_summary_graph, abstract)
            if generated_summary_textrank is not None:
                score_textrank = processing.rouge_score(generated_summary_textrank, abstract)
            else:
                score_textrank = None
            score_dutta = processing.rouge_score(generated_summary_dutta, abstract)
            score_chatterjee = processing.rouge_score(generated_summary_chatterjee, abstract)

            evolutionary_scores.append(score_evolutionary)
            graph_scores.append(score_graphs)
            if score_textrank is not None:
                text_rank_scores.append(score_textrank)
            dutta_scores.append(score_dutta)
            chatterjee_scores.append(score_chatterjee)

    print("RESULTS on the DUC2002 dataset:")

    print("Evolutionary average score:")
    print(processing.final_results(evolutionary_scores))
    print("Graphs average score: ")
    print(processing.final_results(graph_scores))
    print("Textrank average score: [for ONLY " + str(len(text_rank_scores)) + " articles]")
    print(processing.final_results(text_rank_scores))
    print("Dutta average score: ")
    print(processing.final_results(dutta_scores))
    print("Chatterjee average score: ")
    print(processing.final_results(chatterjee_scores))

    print(str(number_of_texts), " articles processing took exactly ", time.time() - start_time, "s")


def test_driver():
    print("TEST benchmark")
    # docs, summaries = duc.get_duc_data()
    # number_of_texts = len(docs)
    # start_time = time.time()
    # for i in range(number_of_texts):
    #     print("current article is " + str(i))
    #     sentences_as_embeddings, text_as_sentences_without_footnotes, abstract, title, title_embedding, rough_abstract = processing.preprocess_duc(
    #         docs[i], duc.get_summary(docs[i], summaries))
    #     if sentences_as_embeddings is not None:
    #

    docs, summaries = duc.get_duc_data()
    number_of_texts = len(docs)

    # evolutionary_scores = []
    graph_scores1 = []
    graph_scores2 = []
    graph_scores3 = []
    graph_scores4 = []
    graph_scores5 = []
    graph_scores6 = []
    graph_scores7 = []

    start_time = time.time()
    for i in range(number_of_texts):
        print("Current article: " + str(i))
        sentences_as_embeddings, text_as_sentences_without_footnotes, abstract, title, title_embedding, rough_abstract = processing.preprocess_duc(
            docs[i], duc.get_summary(docs[i], summaries))
        if sentences_as_embeddings is not None:

            try:
                generated_summary_graph1 = graph.generate_summary_graph(sentences_as_embeddings, text_as_sentences_without_footnotes, processing.number_of_sentences_in_text(abstract), threshold=0.1, cluster_strategy="aslpaw")
                generated_summary_graph2 = graph.generate_summary_graph(sentences_as_embeddings, text_as_sentences_without_footnotes, processing.number_of_sentences_in_text(abstract), threshold=0.1, cluster_strategy="label_propagation")
                generated_summary_graph3 = graph.generate_summary_graph(sentences_as_embeddings, text_as_sentences_without_footnotes, processing.number_of_sentences_in_text(abstract), threshold=0.1, cluster_strategy="greedy_modularity")
                generated_summary_graph4 = graph.generate_summary_graph(sentences_as_embeddings, text_as_sentences_without_footnotes, processing.number_of_sentences_in_text(abstract), threshold=0.1, cluster_strategy="markov_clustering")
                generated_summary_graph5 = graph.generate_summary_graph(sentences_as_embeddings, text_as_sentences_without_footnotes, processing.number_of_sentences_in_text(abstract), threshold=0.1, cluster_strategy="walktrap")
                generated_summary_graph6 = graph.generate_summary_graph(sentences_as_embeddings, text_as_sentences_without_footnotes, processing.number_of_sentences_in_text(abstract), threshold=0.1, cluster_strategy="leiden")
                generated_summary_graph7 = graph.generate_summary_graph(sentences_as_embeddings, text_as_sentences_without_footnotes, processing.number_of_sentences_in_text(abstract), threshold=0.1, cluster_strategy="infomap")


                score_graphs1 = processing.rouge_score(generated_summary_graph1, abstract)
                score_graphs2 = processing.rouge_score(generated_summary_graph2, abstract)
                score_graphs3 = processing.rouge_score(generated_summary_graph3, abstract)
                score_graphs4 = processing.rouge_score(generated_summary_graph4, abstract)
                score_graphs5 = processing.rouge_score(generated_summary_graph5, abstract)
                score_graphs6 = processing.rouge_score(generated_summary_graph6, abstract)
                score_graphs7 = processing.rouge_score(generated_summary_graph7, abstract)


                # print(score_graphs)

                graph_scores1.append(score_graphs1)
                graph_scores2.append(score_graphs2)
                graph_scores3.append(score_graphs3)
                graph_scores4.append(score_graphs4)
                graph_scores5.append(score_graphs5)
                graph_scores6.append(score_graphs6)
                graph_scores7.append(score_graphs7)
            except:
                print("exception at i = " + str(i) + ". skipping.")


    print("RESULTS:")

    # print("Graphs average score: ")
    print(processing.final_results(graph_scores1))
    print(processing.final_results(graph_scores2))
    print(processing.final_results(graph_scores3))
    print(processing.final_results(graph_scores4))
    print(processing.final_results(graph_scores5))
    print(processing.final_results(graph_scores6))
    print(processing.final_results(graph_scores7))

    print(str(number_of_texts), " articles processing took exactly ", time.time() - start_time, "s")


def main():
    option = int(sys.argv[1])
    if option == 1:
        ui_driver()
    elif option == 2:
        theology_driver()
    elif option == 3:
        duc_driver()
    elif option == 4:
        test_driver()
    else:
        print("BAD INPUT!")


main()
