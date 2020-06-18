import sys
from PyQt5.QtWidgets import QMainWindow, QApplication
from Classifier.DecisionTree import*
from GUI.DS_GUI import Start_Window

app = QApplication(sys.argv)
w = Start_Window()
w.show()
app.setStyleSheet("".join(open("style.css").readlines()))
sys.exit(app.exec())
