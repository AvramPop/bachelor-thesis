import sys
from PySide6 import QtCore, QtWidgets, QtGui
from DemoWindow import *


def launch_ui():
    app = QtWidgets.QApplication([])
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.__title = QtWidgets.QLabel("Pop Daniel Avram - B.Sc. Thesis Demo",
                                        alignment=QtCore.Qt.AlignCenter)
        self.__demo_button = QtWidgets.QPushButton("Demo")
        self.__demo_button.clicked.connect(self.__launch_demo_window)

        self.__stats_button = QtWidgets.QPushButton("Stats")
        self.__stats_button.clicked.connect(self.__launch_stats_window)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.__title)
        self.layout.addWidget(self.__demo_button)
        self.layout.addWidget(self.__stats_button)

        self.__demo_window = DemoWindow(["Title 1", "Title 2"])

    @QtCore.Slot()
    def __launch_demo_window(self):
        self.__demo_window.show()

    @QtCore.Slot()
    def __launch_stats_window(self):
        print("stats window")
