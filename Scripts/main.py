#Image Processing
import numpy as np
import matplotlib.pyplot as plt
from skimage import util 
from skimage.color import rgb2gray

#GUI Related Content
import sys
from PyQt5.QtWidgets import *

from PyQt5.QtGui import *

#AI Content
from torchvision.datasets import MNIST
import torch
from torchvision import datasets, transforms
import torchvision.models as models
from torch import nn, optim, cuda
from torch.utils import data
from time import time


#AI PARAMETERS
# epochNum = 
batch_size = 64
# learning_rate = 

device = 'cuda' if cuda.is_available() else 'cpu'

def initAndLoadMNIST():

    trainData = datasets.MNIST(root = 'Data\TrainData', train = True, transform = transforms.ToTensor(), download = True)
    testData = datasets.MNIST(root = 'Data\TestData', train = False, transform = transforms.ToTensor())

    #Load data with transformations
    trainLoader = data.DataLoader(dataset = trainData, batch_size = batch_size, shuffle = True)
    testLoader = data.DataLoader(dataset = testData, batch_size = batch_size, shuffle = False)


    # Showing images

    # dataiter = iter(trainLoader)
    # images, labels = dataiter.next()

    # im2display = images[1].numpy().squeeze()

    # plt.imshow(im2display, interpolation='nearest', cmap='gray_r')
    # plt.show()


class mainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()
        

    # Popup window for train model view
    def trainModelDialog(self):
        
        # Buttons, Labels, and text browser to show progress
        downloadMNIST = QPushButton('Download MNIST', self)
        trainButton = QPushButton('Train', self)
        cancelButton = QPushButton('Cancel', self)
        progressBar = QProgressBar(self)
        progressLabel = QLabel('Progress: ')

        # Creating text box to append download progress status
        msg = QTextBrowser()

        # This is where you append the progress of the download as text
        msg.append('Hello World, this is how you add text to this box')
        
        
        # Dialog configuration and size
        widget = QDialog(self)
        widget.setFixedSize(575, 300)
        widget.setWindowTitle('Dialog')
        widgetGrid = QGridLayout()

        # Grid layout for buttons
        widget.setLayout(widgetGrid)
        widgetGrid.addWidget(msg, 0, 0, 1, 4)
        widgetGrid.addWidget(progressLabel, 1, 0)
        widgetGrid.addWidget(progressBar, 1, 1, 1, 4)
        widgetGrid.addWidget(downloadMNIST, 2, 0)
        widgetGrid.addWidget(trainButton, 2, 1)
        widgetGrid.addWidget(cancelButton, 2, 2)

        widget.exec_()
        

    def openImage(self):
        image = QPixmap('Icons\MNIST.PNG')
        pic = QLabel(self)
        pic.resize(600, 400)
        pic.setScaledContents(True)
        pic.setPixmap(image)
        pic.show()


    def initUI(self):
        initAndLoadMNIST()
        viewTrainingImages = QAction('View Training Images', self)
        viewTestingImages = QAction('View Testing Images', self)

        viewTestingImages.triggered.connect(self.openImage)
        self.setWindowIcon(QIcon('Icons\write.jpg'))


        # Model train GUI section dealing with button presses, new dialog, and progress bar
        trainModelView = QAction('Train Model', self)
        trainModelView.setStatusTip('Train Model')
        trainModelView.triggered.connect(self.trainModelDialog)

        
   
        # File drop down to quit program
        quitProgram = QAction('Quit', self)
        quitProgram.triggered.connect(qApp.quit)
        


        # Add menus and sub-menus to the program menu bar
        menubar = self.menuBar()
        filemenu = menubar.addMenu('&File')
        filemenu.addAction(trainModelView)
        filemenu.addAction(quitProgram)


        # File drop downs for training and testing image viewing




        filemenu = menubar.addMenu('&View')
        filemenu.addAction(viewTrainingImages)
        filemenu.addAction(viewTestingImages)
        self.setWindowTitle('Handwritten Digit Recogniser')

        # Configure size of window
        self.statusBar()
        self.resize(600, 400)
        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    ex = mainWindow()
    sys.exit(app.exec_())




#Note all methods below need to be adjusted for scope of variables and may need to be removed from methods



def setModel():
    model = models.resnet18(pretrained = False, progress = True)

# def trainModel(INSERT PARAMETERS):
    #INSERT CODE HERE
    #Needs optimiser, 

# def testModel(INSERT PARAMETERS):
    #INSERT CODE HERE

# def ProcessAndRecogniseNumber();
    #Insert COde HEre
    #Note this can be split into different methods
    #Code that puts turns input into image, processes as same way as TEST/TRAIN SETS
    #LOADS INTO NEW LOADER
    #Passes image to AI
    #Returns ai output
    #Shows probability?
