from evo import evolutionary, chatterjee
from graphs import graph, dutta, textrank
import processing.processing_utils as processing
import time
import ui.ui_main as ui
from processing import duc
import sys


def ui_driver():
    titles = processing.get_titles(processing.get_number_of_texts_in_folder('/home/dani/Desktop/licenta/bachelor-thesis/thesis-project/resources/articles'))
    ui.launch_ui(titles)


def theology_driver(number_of_texts=49):
    print("Theology benchmark")
    evolutionary_scores = []
    graphs_scores = []
    text_rank_scores = []
    dutta_scores = []
    chatterjee_scores = []
    count1 = 0
    count2 = 0
    count3 = 0
    start_time = time.time()
    for i in range(1, number_of_texts + 1):
        print("Current article: " + str(i))
        sentences_as_embeddings, text_as_sentences_without_footnotes, abstract, title, title_embedding, rough_abstract = processing.prepare_data(i)

        summary_length = processing.number_of_sentences_in_text(abstract)
        generated_summary_evolutionary = evolutionary.generate_summary_evolutionary(sentences_as_embeddings, title_embedding, text_as_sentences_without_footnotes,
                                                                                    summary_length, a=0.2, b=0.2, c=0.2, d=0.2, e=0.2)
        try:
            generated_summary_graphs = graph.generate_summary_graph(sentences_as_embeddings, text_as_sentences_without_footnotes, summary_length,
                                                                    cluster_strategy="leiden", threshold=0.55)
            score_graphs = processing.rouge_score(generated_summary_graphs, abstract)
            graphs_scores.append(score_graphs)
        except:
            print("Exception")
        else:
            count1 = count1 + 1

        try:
            generated_summary_textrank = textrank.generate_summary_graph_text_rank(text_as_sentences_without_footnotes, summary_length)
            score_textrank = processing.rouge_score(generated_summary_textrank, abstract)
            text_rank_scores.append(score_textrank)
        except:
            print("Exception")
        else:
            count2 = count2 + 1

        try:
            generated_summary_dutta = dutta.generate_summary_dutta(text_as_sentences_without_footnotes, summary_length)
            score_dutta = processing.rouge_score(generated_summary_dutta, abstract)
            dutta_scores.append(score_dutta)
        except:
            print("Exception")
        else:
            count3 = count3 + 1

        generated_summary_chatterjee = chatterjee.generate_summary_chatterjee(text_as_sentences_without_footnotes, summary_length)

        score_evolutionary = processing.rouge_score(generated_summary_evolutionary, abstract)
        score_chatterjee = processing.rouge_score(generated_summary_chatterjee, abstract)

        evolutionary_scores.append(score_evolutionary)
        chatterjee_scores.append(score_chatterjee)

        print(evolutionary_scores)
        print(graphs_scores)
        print(text_rank_scores)
        print(dutta_scores)
        print(chatterjee_scores)

    print("RESULTS on the Theology dataset:")

    print("Evolutionary average score:")
    print(processing.final_results(evolutionary_scores))
    print("Graphs average score: [for ONLY" + str(count1) + " articles]")
    print(processing.final_results(graphs_scores))
    print("Textrank average score: [for ONLY " + str(count2) + " articles]")
    print(processing.final_results(text_rank_scores))
    print("Dutta average score: [for ONLY" + str(count3) + "articles ")
    print(processing.final_results(dutta_scores))
    print("Chatterjee average score: ")
    print(processing.final_results(chatterjee_scores))

    print(str(number_of_texts), " articles processing took exactly ", time.time() - start_time, "s")


def duc_driver():
    print("DUC benchmark")
    docs, summaries = duc.get_duc_data()
    number_of_texts = len(docs)
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
                                                                   summary_length, cluster_strategy="leiden", threshold=0.05)
            generated_summary_textrank = textrank.generate_summary_graph_text_rank(text_as_sentences_without_footnotes, summary_length)
            generated_summary_dutta = dutta.generate_summary_dutta(text_as_sentences_without_footnotes, summary_length, threshold=0.05)
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


def main():
    option = int(sys.argv[1])
    if option == 1:
        ui_driver()
    elif option == 2:
        theology_driver()
    elif option == 3:
        duc_driver()
    else:
        print("BAD INPUT!")


main()
