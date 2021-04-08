#Image Processing
import numpy as py
import matplotlib.pyplot as plt

#GUI Related Content
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, qApp
from PyQt5.QtGui import QIcon, QPixmap

#AI Content
from torchvision.datasets import MNIST
import torch
from torchvision import datasets, transforms
import torchvision.models as models
from torch import nn, optim

#AI PARAMETERS
epochNum = 
batch_size = 
learning_rate = 

class MyApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initAndLoadMNIST():
        datasetTransform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5),(0.5))])
        TRAIN = datasets.MNIST(root ='INSERT DIRECTORY', train = True, transform = datasetTransform, download = True)
        TEST = datasets.MNIST(root ='INSERT DIRECTORY', train = False, transform = datasetTransform, download = True)
        #Load data with transformations
        TESTLOADER = torch.utils.data.DataLoader(TEST, batch_size=batch_size)
        TRAINLOADER = torch.utils.data.DataLoader(TRAIN, batch_size=batch_size)


    def initUI(self):
        self.setWindowIcon(QIcon('Icons\write.jpg'))
        self.setGeometry(300, 300, 300, 200)

        # File drop down to train model
        trainModelView = QAction('Train Model', self)
        trainModelView.setStatusTip('Train Model')


        # File drop down to quit program
        quitProgram = QAction('Quit', self)
        quitProgram.triggered.connect(qApp.quit)

        # File drop downs for training and testing image viewing
        viewTrainingImages = QAction('View Training Images', self)
        viewTestingImages = QAction('View Testing Images', self)

        self.statusBar()


        # Add menus and sub-menus to the program menu bar
        menubar = self.menuBar()
        filemenu = menubar.addMenu('&File')
        filemenu.addAction(trainModelView)
        filemenu.addAction(quitProgram)

        filemenu = menubar.addMenu('&View')
        filemenu.addAction(viewTrainingImages)
        filemenu.addAction(viewTestingImages)

        self.setWindowTitle('Handwritten Digit Recogniser')
        self.setGeometry(300, 300, 300, 200)

        # Configure size of window
        self.resize(600, 400)
        self.show()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())

