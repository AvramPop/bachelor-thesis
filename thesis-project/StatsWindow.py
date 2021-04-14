from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtCore import (Signal, QMutex, QMutexLocker, QPointF, QSize, Qt,
        QThread, QWaitCondition)
import processing
import evolutionary
import graph


class StatsWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
