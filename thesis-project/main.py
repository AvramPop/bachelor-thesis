from evo import evolutionary, chatterjee
from graphs import graph, dutta, textrank
import processing.processing_utils as processing
import time
import ui.ui_main as ui
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


def test_driver_1(number_of_texts=39):
    print("Theology evo")
    evolutionary_scores_1 = []
    evolutionary_scores_2 = []
    evolutionary_scores_3 = []
    evolutionary_scores_4 = []
    evolutionary_scores_5 = []
    start_time = time.time()
    for i in range(1, number_of_texts + 1):
        print("Current article: " + str(i))
        sentences_as_embeddings, text_as_sentences_without_footnotes, abstract, title, title_embedding, rough_abstract = processing.prepare_data(
            i)

        summary_length = processing.number_of_sentences_in_text(abstract)
        generated_summary_evolutionary_1 = evolutionary.generate_summary_evolutionary(sentences_as_embeddings,
                                                                                    title_embedding,
                                                                                    text_as_sentences_without_footnotes,
                                                                                    summary_length, a=0.2, b=0.2, c=0.2, d=0.2, e=0.2)
        generated_summary_evolutionary_2 = evolutionary.generate_summary_evolutionary(sentences_as_embeddings,
                                                                                    title_embedding,
                                                                                    text_as_sentences_without_footnotes,
                                                                                    summary_length, a=0.1, b=0.1, c=0.25, d=0.3, e=0.25)
        generated_summary_evolutionary_3 = evolutionary.generate_summary_evolutionary(sentences_as_embeddings,
                                                                                    title_embedding,
                                                                                    text_as_sentences_without_footnotes,
                                                                                    summary_length, a=0.3, b=0.3, c=0.1, d=0.2, e=0.1)
        generated_summary_evolutionary_4 = evolutionary.generate_summary_evolutionary(sentences_as_embeddings,
                                                                                    title_embedding,
                                                                                    text_as_sentences_without_footnotes,
                                                                                    summary_length, a=0.25, b=0.25, c=0.1, d=0.3, e=0.1)
        generated_summary_evolutionary_5 = evolutionary.generate_summary_evolutionary(sentences_as_embeddings,
                                                                                    title_embedding,
                                                                                    text_as_sentences_without_footnotes,
                                                                                    summary_length, a=0.1, b=0.3, c=0.1, d=0.2, e=0.3)

        score_evolutionary_1 = processing.rouge_score(generated_summary_evolutionary_1, abstract)
        score_evolutionary_2 = processing.rouge_score(generated_summary_evolutionary_2, abstract)
        score_evolutionary_3 = processing.rouge_score(generated_summary_evolutionary_3, abstract)
        score_evolutionary_4 = processing.rouge_score(generated_summary_evolutionary_4, abstract)
        score_evolutionary_5 = processing.rouge_score(generated_summary_evolutionary_5, abstract)

        evolutionary_scores_1.append(score_evolutionary_1)
        evolutionary_scores_2.append(score_evolutionary_2)
        evolutionary_scores_3.append(score_evolutionary_3)
        evolutionary_scores_4.append(score_evolutionary_4)
        evolutionary_scores_5.append(score_evolutionary_5)

    print("RESULTS on the Theology dataset:")

    print("Evolutionary 1:")
    print(processing.final_results(evolutionary_scores_1))
    print("Evolutionary 2:")
    print(processing.final_results(evolutionary_scores_2))
    print("Evolutionary 3:")
    print(processing.final_results(evolutionary_scores_3))
    print("Evolutionary 4:")
    print(processing.final_results(evolutionary_scores_4))
    print("Evolutionary 5:")
    print(processing.final_results(evolutionary_scores_5))

    print(str(number_of_texts), " articles processing took exactly ", time.time() - start_time, "s")


def main():
    option = int(sys.argv[1])
    if option == 1:
        ui_driver()
    elif option == 2:
        theology_driver(2)
    elif option == 3:
        duc_driver()
    elif option == 4:
        test_driver_1()
    else:
        print("BAD INPUT!")


main()
