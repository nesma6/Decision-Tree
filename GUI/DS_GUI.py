import PyQt5
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5 import QtWidgets
import os.path
from PyQt5.QtWidgets import*
from PyQt5.QtGui import*
from PyQt5.QtCore import*
import sys
import os
path = os.curdir.replace('\GUI',"")
sys.path.append(path)
path = sys.path[-1]
from Classifier.DecisionTree import*


class Start_Window(QMainWindow):

    def __init__(DStart):
        super(Start_Window,DStart).__init__()
        DStart.title = 'Decision Tree Classifier'

        DStart.InitWindow()
        DStart.Design()
        DStart.setObjectNames()
        DStart.btnTrain.clicked.connect(DStart.onTrainClicked)
        DStart.btnPredict.clicked.connect(DStart.onPredictClicked)
        DStart.btnShow.clicked.connect(DStart.onShowClicked)
        DStart.btnBack.clicked.connect(DStart.onBackClicked)
        DStart.depthEdit.textChanged.connect(DStart.changeStatus)
        DStart.btnClassify.clicked.connect(DStart.onClassifyClicked)

    def InitWindow(DStart):
        DStart.setWindowTitle(DStart.title)
        # DStart.setAutoFillBackground(True)

        p = DStart.palette()
        p.setColor(DStart.backgroundRole(), Qt.white) #OR teleStart.setStyleSheet("background-color: white;")
        DStart.setPalette(p)

        DStart.mainWidget = QWidget(DStart)
        DStart.mainLayout = QGridLayout()

        #-------------------1st Left Widget--------------------------------------
        #-------------------Upper GroupBox----------------------------
        DStart.stackLay = QStackedLayout()
        DStart.w1 = QWidget()
        DStart.lay1 = QGridLayout()
        DStart.trainBox = QGroupBox("Training Decision Tree")
        DStart.layTrainBox = QVBoxLayout()
        DStart.layTrain = QFormLayout()
        DStart.lblDepth = QLabel("Decision Tree Maximum Depth:")
        DStart.depthEdit  = QLineEdit()
        DStart.depthEdit.setPlaceholderText("maximum depth")
        DStart.btnTrain = QPushButton("Train Decision Tree")
        DStart.lblAccuracy = QLabel(" ")

        #-----------------Bottom GroupBox-------------------------------
        DStart.classifyBox = QGroupBox("Reviews Classification")
        DStart.layClassifyBox = QGridLayout()
        DStart.reviewText = QTextEdit()
        DStart.reviewText.setPlaceholderText("review text")
        DStart.btnClassify = QPushButton("Classify Review")
        DStart.btnClassify.setMaximumWidth(200)


        DStart.fileEdit = QLineEdit()
        DStart.fileEdit.setPlaceholderText("testFileName.csv")
        DStart.btnPredict = QPushButton("Predict Reviews")
        DStart.lblClassification = QLabel("")

        #----------------2nd Left Widget-----------------------------
        DStart.predictBox = QGroupBox("Review Prediciton")
        DStart.layPredictBox = QVBoxLayout()
        DStart.w2 = QWidget()
        DStart.layW2 = QVBoxLayout()
        DStart.btnBack = QPushButton("Back")
        DStart.tablePrediction = QTableWidget()
        DStart.lblGraph = QLabel()

        #----------------Right Widget---------------------------------
        DStart.gBox1 = QGroupBox("Decision Tree Classifier")
        DStart.gBox1 .setAlignment(Qt.AlignCenter)
        DStart.lay2 = QVBoxLayout()
        DStart.scrollarea = QScrollArea()
        DStart.gBox2 = QGroupBox("Decision Tree")
        DStart.gBox2 .setAlignment(Qt.AlignCenter)
        DStart.btnShow = QPushButton("Show Predictions")
        




    def Design(DStart):
        qtRectangle = DStart.frameGeometry()
        centerPoint = QDesktopWidget().availableGeometry().center()
        qtRectangle.moveCenter(centerPoint)

        DStart.setCentralWidget(DStart.mainWidget)
        DStart.mainWidget.setLayout(DStart.mainLayout)
        DStart.mainLayout.addLayout(DStart.stackLay,0,0)
        DStart.mainLayout.setColumnStretch(0,1)
        DStart.mainLayout.setColumnStretch(1,2)
        dummylay2 = QHBoxLayout()
        DStart.mainLayout.addLayout(dummylay2,0,1)
        dummylay2.addWidget(DStart.scrollarea)
        DStart.scrollarea.setWidget(DStart.gBox2)
        DStart.scrollarea.setWidgetResizable(True)
        
        


        DStart.stackLay.addWidget(DStart.w1)
        DStart.stackLay.addWidget(DStart.w2)
        DStart.w1.setLayout(DStart.lay1)
        DStart.lay1.addWidget(DStart.trainBox)
        DStart.lay1.addWidget(DStart.classifyBox)
        DStart.lay1.setRowStretch(0,1)
        DStart.lay1.setRowStretch(1,1)
        DStart.stackLay.setCurrentIndex(0)
        DStart.gBox2.setLayout(DStart.lay2)

        DStart.trainBox.setLayout(DStart.layTrainBox)
        DStart.layTrainBox.addLayout(DStart.layTrain)
        

        DStart.layTrain.addRow(DStart.lblDepth,DStart.depthEdit)
        

        dummyLay = QHBoxLayout()
        dummyLay.addWidget(DStart.btnTrain)
        dummyLay.setAlignment(Qt.AlignRight)
        DStart.layTrain.setAlignment(Qt.AlignCenter)
        DStart.layTrain.setFormAlignment(Qt.AlignCenter)

        DStart.layTrain.addRow(dummyLay)
        DStart.layTrain.addWidget(DStart.lblAccuracy)

        DStart.classifyBox.setLayout(DStart.layClassifyBox)
        DStart.layClassifyBox.addWidget(DStart.reviewText,0,0)
        dummyClassify = QGridLayout()
        dummyClassify.addWidget(DStart.btnClassify,0,0)
        dummyClassify.addWidget(DStart.lblClassification,1,0)
        DStart.layClassifyBox.addLayout(dummyClassify,0,1)
        # DStart.layClassifyBox.addWidget(DStart.btnClassify,0,1)
        DStart.layClassifyBox.addWidget(DStart.fileEdit,1,0)
        DStart.layClassifyBox.addWidget(DStart.btnPredict,1,1)
        DStart.layClassifyBox.addWidget(DStart.btnShow,2,1)
        DStart.layClassifyBox.setColumnStretch(0,2)
        DStart.layClassifyBox.setColumnStretch(1,1)
        DStart.layClassifyBox.setRowStretch(0,1)
        DStart.layClassifyBox.setRowStretch(1,2)
        DStart.layClassifyBox.setAlignment(Qt.AlignCenter)
        
        DStart.w2.setLayout(DStart.layW2)
        dummylay3 = QHBoxLayout()
        dummylay3.addWidget(DStart.btnBack)
        dummylay3.setAlignment(Qt.AlignCenter)
        DStart.layW2.addWidget(DStart.tablePrediction)
        DStart.tablePrediction.setColumnCount(2)
        DStart.tablePrediction.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        DStart.tablePrediction.setHorizontalHeaderLabels(["Reviews Text","Prediction"])
        DStart.layW2.addLayout(dummylay3)
        

        DStart.lay2.addWidget(DStart.lblGraph)
        # DStart.show()


    def setObjectNames(DStart):
        DStart.gBox1.setObjectName("TrainBox")
        DStart.btnShow.setObjectName("Back")


    @pyqtSlot()
    def changeStatus(DStart):
        DStart.statusBar().showMessage("Decision Tree is ready to be trained")
    def onTrainClicked(DStart):
        # Call the decision_tree_algorithm train the model and draw the output graph
        DStart.statusBar().showMessage("Decision Tree is being trained..")
        dataFrame = data_init(path+"/Datasets/sample_train.csv")
        adjustedDataframe = words_count(dataFrame)
        depth = int(DStart.depthEdit.text())
        DStart.tree = decision_tree(adjustedDataframe,max_depth=depth)
        
        d.clear()
        draw_graph(DStart.tree.root)
        d.render(format='png')
        DStart.graph = QPixmap('graph.gv.png')
        DStart.lblGraph.setPixmap(DStart.graph)
        dev = data_init(path+"/Datasets/sample_dev.csv")
        DStart.lblAccuracy.setText("Accuracy:\n"+str(round(calculate_accuracy(dev,DStart.tree)*100,2))+"%")
        DStart.statusBar().showMessage("Decision Tree is trained")
    
    def onPredictClicked(DStart):
        # DStart.statusBar().showMessage("Review is ")
        name,_ = QtWidgets.QFileDialog.getOpenFileName(DStart,'Open File')
        file = open(name,'r')

        with file:
            DStart.fileEdit.setText(file.name)
            DStart.testData = data_init(file.name)
            DStart.p = predict(DStart.testData,DStart.tree)
    
    def onShowClicked(DStart):
        DStart.stackLay.setCurrentIndex(1)
        DStart.fillTable()
    
    def onBackClicked(DStart):
        DStart.stackLay.setCurrentIndex(0)
    
    def onClassifyClicked(DStart):
        DStart.lblClassification.setText(classify_review_text(DStart.reviewText.toPlainText(),DStart.tree.root))



    def fillTable(DStart):
        with open('output.txt', 'r') as f:
            DStart.tablePrediction.setRowCount(len(DStart.p))
            for i in range (0,len(DStart.p)):
                DStart.tablePrediction.setItem(i, 0, QTableWidgetItem(DStart.testData["reviews.text"][i]))
                DStart.tablePrediction.setItem(i, 1, QTableWidgetItem(f.readline()))
        DStart.tablePrediction.resizeRowsToContents()
