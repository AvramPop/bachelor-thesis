from PySide6 import QtCore, QtWidgets
import processing
import evolutionary
import graph


class StatsWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QtWidgets.QVBoxLayout(self)

        self.__setup_number_ui()
        self.__setup_titles_ui()
        self.__setup_params_ui()
        self.__setup_stats_ui()
        self.__setup_graph_ui()

    def __setup_number_ui(self):
        container = QtWidgets.QHBoxLayout()

        container.addWidget(QtWidgets.QLabel("Number of articles to analyze: "))
        self.__number_input = QtWidgets.QLineEdit()
        self.__number_input.setFixedWidth(120)
        container.addWidget(self.__number_input)

        self.layout.addLayout(container)

    def __setup_titles_ui(self):
        container = QtWidgets.QHBoxLayout()

        container.addWidget(QtWidgets.QLabel("Graphs algorithm metaparameters"))
        container.addWidget(QtWidgets.QLabel("Evolutionary algorithm metaparameters"))
        self.layout.addLayout(container)

    def __setup_params_ui(self):
        container = QtWidgets.QHBoxLayout()

        graphs_container = QtWidgets.QVBoxLayout()
        threshold_container = QtWidgets.QHBoxLayout()
        threshold_container.addWidget(QtWidgets.QLabel("Cosine similarity threshold: "))
        self.__similarity_threshold_input = QtWidgets.QLineEdit()
        self.__similarity_threshold_input.setFixedWidth(120)
        threshold_container.addWidget(self.__similarity_threshold_input)
        graphs_container.addLayout(threshold_container)

        algorithm_container = QtWidgets.QHBoxLayout()
        algorithm_container.addWidget(QtWidgets.QLabel("Clustering algorithm: "))
        self.__clustering_algorithm_input = QtWidgets.QComboBox()
        self.__clustering_algorithm_input.addItem("aslpaw")
        self.__clustering_algorithm_input.addItem("label_propagation")
        self.__clustering_algorithm_input.addItem("greedy_modularity")
        self.__clustering_algorithm_input.addItem("markov_clustering")
        self.__clustering_algorithm_input.addItem("walktrap")
        self.__clustering_algorithm_input.addItem("leiden")
        self.__clustering_algorithm_input.addItem("infomap")
        algorithm_container.addWidget(self.__clustering_algorithm_input)
        graphs_container.addLayout(algorithm_container)

        evolutionary_container = QtWidgets.QVBoxLayout()

        iters_container = QtWidgets.QHBoxLayout()
        iters_container.addWidget(QtWidgets.QLabel("Number of iterations: "))
        self.__iters_input = QtWidgets.QLineEdit()
        self.__iters_input.setFixedWidth(120)
        iters_container.addWidget(self.__iters_input)
        evolutionary_container.addLayout(iters_container)

        pop_size_container = QtWidgets.QHBoxLayout()
        pop_size_container.addWidget(QtWidgets.QLabel("Population size: "))
        self.__pop_size_input = QtWidgets.QLineEdit()
        self.__pop_size_input.setFixedWidth(120)
        pop_size_container.addWidget(self.__pop_size_input)
        evolutionary_container.addLayout(pop_size_container)

        cohesion_container = QtWidgets.QHBoxLayout()
        cohesion_container.addWidget(QtWidgets.QLabel("Cohesion coefficient: "))
        self.__cohesion_input = QtWidgets.QLineEdit()
        self.__cohesion_input.setFixedWidth(120)
        cohesion_container.addWidget(self.__cohesion_input)
        evolutionary_container.addLayout(cohesion_container)

        readability_container = QtWidgets.QHBoxLayout()
        readability_container.addWidget(QtWidgets.QLabel("Readability coefficient: "))
        self.__readability_input = QtWidgets.QLineEdit()
        self.__readability_input.setFixedWidth(120)
        readability_container.addWidget(self.__readability_input)
        evolutionary_container.addLayout(readability_container)

        sentence_position_container = QtWidgets.QHBoxLayout()
        sentence_position_container.addWidget(QtWidgets.QLabel("Sentence position coefficient: "))
        self.__sentence_input = QtWidgets.QLineEdit()
        self.__sentence_input.setFixedWidth(120)
        sentence_position_container.addWidget(self.__sentence_input)
        evolutionary_container.addLayout(sentence_position_container)

        title_container = QtWidgets.QHBoxLayout()
        title_container.addWidget(QtWidgets.QLabel("Title relation coefficient: "))
        self.__title_input = QtWidgets.QLineEdit()
        self.__title_input.setFixedWidth(120)
        title_container.addWidget(self.__title_input)
        evolutionary_container.addLayout(title_container)

        length_container = QtWidgets.QHBoxLayout()
        length_container.addWidget(QtWidgets.QLabel("Sentence length coefficient: "))
        self.__length_input = QtWidgets.QLineEdit()
        self.__length_input.setFixedWidth(120)
        length_container.addWidget(self.__length_input)
        evolutionary_container.addLayout(length_container)

        container.addLayout(graphs_container)
        container.addLayout(evolutionary_container)

        self.layout.addLayout(container)

    def __setup_stats_ui(self):
        container = QtWidgets.QVBoxLayout()

        stats_button = QtWidgets.QPushButton("Get statistics")
        stats_button.clicked.connect(self.__generate_statistics)
        container.addWidget(stats_button)

        graphs_container = QtWidgets.QVBoxLayout()
        graphs_container.addWidget(QtWidgets.QLabel("Graphs statistics"))
        self.__graphs_rouge_1_f_label = QtWidgets.QLabel("")
        graphs_container.addWidget(self.__graphs_rouge_1_f_label)
        container.addLayout(graphs_container)

        evolutionary_container = QtWidgets.QVBoxLayout()
        evolutionary_container.addWidget(QtWidgets.QLabel("Evolutionary statistics"))
        self.__evolutionary_rouge_1_f_label = QtWidgets.QLabel("")
        evolutionary_container.addWidget(self.__evolutionary_rouge_1_f_label)
        container.addLayout(evolutionary_container)

        self.layout.addLayout(container)

    @QtCore.Slot()
    def __generate_statistics(self):
        number_of_texts = int(self.__number_input.text())
        evolutionary_scores = []
        graph_scores = []
        for i in range(1, number_of_texts + 1):
            sentences_as_embeddings, text_as_sentences_without_footnotes, abstract, title, title_embedding, rough_abstract = processing.prepare_data(i)

            generated_summary_evolutionary = \
                evolutionary.generate_summary_evolutionary(sentences_as_embeddings,
                                                           title_embedding,
                                                           text_as_sentences_without_footnotes,
                                                           processing.number_of_sentences_in_text(abstract),
                                                           number_of_iterations=int(self.__iters_input.text()),
                                                           population_size=int(self.__pop_size_input.text()),
                                                           a=float(self.__cohesion_input.text()),
                                                           b=float(self.__readability_input.text()),
                                                           c=float(self.__sentence_input.text()),
                                                           d=float(self.__title_input.text()),
                                                           e=float(self.__length_input.text()))
            generated_summary_graph = graph.generate_summary_graph(sentences_as_embeddings, text_as_sentences_without_footnotes, processing.number_of_sentences_in_text(abstract), cluster_strategy=self.__clustering_algorithm_input.currentText(), threshold=float(self.__similarity_threshold_input.text()))

            score_evolutionary = processing.rouge_score(generated_summary_evolutionary, abstract)
            score_graphs = processing.rouge_score(generated_summary_graph, abstract)

            evolutionary_scores.append(score_evolutionary)
            graph_scores.append(score_graphs)

        self.__evolutionary_rouge_1_f_label.setText("ROUGE-1-F: " + str(processing.final_results(evolutionary_scores)["ROUGE-1-F"]))
        self.__graphs_rouge_1_f_label.setText("ROUGE-1-F: " + str(processing.final_results(graph_scores)["ROUGE-1-F"]))

    def __setup_graph_ui(self):
        container = QtWidgets.QVBoxLayout()

        plots_button = QtWidgets.QPushButton("Generate plots")
        plots_button.clicked.connect(self.__generate_plots)
        container.addWidget(plots_button)

        self.layout.addLayout(container)

    @QtCore.Slot()
    def __generate_plots(self):
        print("plots")
