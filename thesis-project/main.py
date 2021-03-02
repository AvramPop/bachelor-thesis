import evolutionary
import graph
import preprocessing


def main():
    sentences_as_embeddings, text_as_sentences_without_footnotes, abstract, title = preprocessing.prepare_data(1)
    print("text length is: " + str(len(text_as_sentences_without_footnotes)))

    generated_summary_evolutionary = evolutionary.generate_summary_evolutionary(sentences_as_embeddings, text_as_sentences_without_footnotes, preprocessing.number_of_sentences_in_text(abstract))
    generated_summary_graph = graph.generate_summary_graph(sentences_as_embeddings, text_as_sentences_without_footnotes, preprocessing.number_of_sentences_in_text(abstract))

    score_evolutionary = preprocessing.rouge_score(generated_summary_evolutionary, abstract)
    score_graphs = preprocessing.rouge_score(generated_summary_graph, abstract)

    print(score_evolutionary)
    print(score_graphs)


main()
