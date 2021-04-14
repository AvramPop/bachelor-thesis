from PySide6 import QtCore, QtWidgets, QtGui
import processing
import evolutionary
import graph


class DemoWindow(QtWidgets.QWidget):
    def __init__(self, titles):
        super().__init__()
        self.__titles = titles

        self.layout = QtWidgets.QHBoxLayout(self)

        self.__setup_articles_ui()
        self.__setup_buttons_ui()
        self.__setup_summary_ui()

    def __setup_summary_ui(self):
        self.__automatic_summary_box = QtWidgets.QVBoxLayout()
        self.__manual_summary_box = QtWidgets.QVBoxLayout()
        container = QtWidgets.QHBoxLayout()

        self.__summary_text = QtWidgets.QTextEdit("")
        self.__summary_text.hide()
        self.__automatic_summary_label = QtWidgets.QLabel("Automatic summary:")
        self.__automatic_summary_label.hide()
        self.__automatic_summary_box.addWidget(self.__automatic_summary_label)
        self.__automatic_summary_box.addWidget(self.__summary_text)

        self.__abstract_text = QtWidgets.QTextEdit("")
        self.__abstract_text.hide()
        self.__manual_abstract_label = QtWidgets.QLabel("Human-written abstract:")
        self.__manual_abstract_label.hide()
        self.__manual_summary_box.addWidget(self.__manual_abstract_label)
        self.__manual_summary_box.addWidget(self.__abstract_text)

        container.addLayout(self.__automatic_summary_box)
        container.addLayout(self.__manual_summary_box)
        self.layout.addLayout(container)

    def __setup_buttons_ui(self):
        buttons_box = QtWidgets.QVBoxLayout()

        graphs_button = QtWidgets.QPushButton("Graphs Summary")
        graphs_button.clicked.connect(self.__get_graphs_abstract)

        evolutionary_button = QtWidgets.QPushButton("Evolutionary Summary")
        evolutionary_button.clicked.connect(self.__get_evolutionary_abstract)

        self.__movie_label = QtWidgets.QLabel()

        buttons_box.addWidget(graphs_button)
        buttons_box.addWidget(self.__movie_label)
        buttons_box.addWidget(evolutionary_button)

        self.layout.addLayout(buttons_box)

    def __setup_articles_ui(self):
        titles_box = QtWidgets.QVBoxLayout()

        articles_label = QtWidgets.QLabel("Articles:")
        self.__titles_list = QtWidgets.QListWidget()
        for title in self.__titles:
            widget_item = QtWidgets.QListWidgetItem()
            widget_item.setText(title)
            self.__titles_list.addItem(widget_item)

        titles_box.addWidget(articles_label)
        titles_box.addWidget(self.__titles_list)

        self.layout.addLayout(titles_box)

    @QtCore.Slot()
    def __get_evolutionary_abstract(self):
        self.__start_wheel()
        article_index = self.__titles_list.currentRow() + 1
        sentences_as_embeddings, text_as_sentences_without_footnotes, abstract, title, title_embedding, rough_abstract = processing.prepare_data(article_index)
        generated_summary_evolutionary = evolutionary.generate_summary_evolutionary(sentences_as_embeddings,
                                                                                    title_embedding,
                                                                                    text_as_sentences_without_footnotes,
                                                                                    processing.number_of_sentences_in_text(
                                                                                        abstract))
        self.__stop_wheel()
        self.__summary_text.setText(generated_summary_evolutionary)
        self.__summary_text.show()
        self.__abstract_text.setText(rough_abstract)
        self.__abstract_text.show()
        self.__automatic_summary_label.show()
        self.__manual_abstract_label.show()

    @QtCore.Slot()
    def __get_graphs_abstract(self):
        article_index = self.__titles_list.currentRow() + 1
        sentences_as_embeddings, text_as_sentences_without_footnotes, abstract, title, title_embedding, rough_abstract = processing.prepare_data(
            article_index)
        generated_summary_graph = graph.generate_summary_graph(sentences_as_embeddings,
                                                               text_as_sentences_without_footnotes,
                                                               processing.number_of_sentences_in_text(abstract))
        self.__stop_wheel()
        self.__summary_text.setText(generated_summary_graph)
        self.__summary_text.show()
        self.__abstract_text.setText(rough_abstract)
        self.__abstract_text.show()
        self.__automatic_summary_label.show()
        self.__manual_abstract_label.show()

    def __start_wheel(self):
        wheel_movie = QtGui.QMovie("resources/ajax-loader.gif")
        self.__movie_label.setMovie(wheel_movie)
        # self.__movie_label.show()
        wheel_movie.start()

    def __stop_wheel(self):
        self.__movie_label.hide()




