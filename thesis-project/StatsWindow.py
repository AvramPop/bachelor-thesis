from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtCore import (Signal, QMutex, QMutexLocker, QPointF, QSize, Qt,
        QThread, QWaitCondition)
import processing
import evolutionary
import graph


class StatsWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QtWidgets.QVBoxLayout(self)

        self.__setup_number_ui()
        self.__setup_params_ui()
        # self.__setup_stats_ui()
        # self.__setup_graph_ui()

    def __setup_number_ui(self):
        container = QtWidgets.QHBoxLayout()

        container.addWidget(QtWidgets.QLabel("Number of articles to analyze: "))
        self.__number_input = QtWidgets.QLineEdit()
        self.__number_input.setFixedWidth(120)
        container.addWidget(self.__number_input)

        self.layout.addLayout(container)

    def __setup_params_ui(self):
        container = QtWidgets.QHBoxLayout()

        graphs_container = QtWidgets.QVBoxLayout()
        graphs_container.addWidget(QtWidgets.QLabel("Graphs algorithm metaparameters"))
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
        evolutionary_container.addWidget(QtWidgets.QLabel("Evolutionary algorithm metaparameters"))

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

