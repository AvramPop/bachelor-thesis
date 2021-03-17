import evolutionary
import graph
import preprocessing
from numpy import average
import time


def main():
    number_of_texts = 1
    evolutionary_scores = []
    graph_scores = []
    start_time = time.time()
    for i in range(1, number_of_texts + 1):
        print("Current article: " + str(i))
        sentences_as_embeddings, text_as_sentences_without_footnotes, abstract, title, title_embedding = preprocessing.prepare_data(i)
        print("text length is: " + str(len(text_as_sentences_without_footnotes)))

        generated_summary_evolutionary = evolutionary.generate_summary_evolutionary(sentences_as_embeddings, title_embedding, text_as_sentences_without_footnotes, preprocessing.number_of_sentences_in_text(abstract))
        generated_summary_graph = graph.generate_summary_graph(sentences_as_embeddings, text_as_sentences_without_footnotes, preprocessing.number_of_sentences_in_text(abstract))

        score_evolutionary = preprocessing.rouge_score(generated_summary_evolutionary, abstract)
        score_graphs = preprocessing.rouge_score(generated_summary_graph, abstract)

        evolutionary_scores.append(score_evolutionary["ROUGE-1-R"])
        graph_scores.append(score_graphs["ROUGE-1-R"])
        print(score_evolutionary)
        print(score_graphs)

    print("Evolutionary average score: " + str(average(evolutionary_scores)))
    print("Graphs average score: " + str(average(graph_scores)))
    print(str(number_of_texts), " articles processing took exactly ", time.time() - start_time, "s")


main()
