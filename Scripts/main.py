#Image Processing
import numpy as np
import matplotlib.pyplot as plt
from skimage import util 
from skimage.color import rgb2gray

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
# epochNum = 
# batch_size = 
# learning_rate = 

def initAndLoadMNIST():

    datasetTransform = transforms.Compose([
        # transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    TRAIN = datasets.MNIST(root = 'Data\TrainData', train = True, transform = datasetTransform, download = False)
    TEST = datasets.MNIST(root = 'Data\TestData', train = False, transform = datasetTransform, download = False)

    #Load data with transformations
    TESTLOADER = torch.utils.data.DataLoader(TEST, batch_size=64)
    TRAINLOADER = torch.utils.data.DataLoader(TRAIN, batch_size=64)


    # SHowing images

    # dataiter = iter(TRAINLOADER)
    # images, labels = dataiter.next()

    # im2display = images[1].numpy().squeeze().transpose((1,2,0))
    # invertedImage = util.invert(im2display)

    # plt.imshow(invertedImage, interpolation='nearest', cmap='gray_r')
    # plt.show()


class MyApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()

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

        initAndLoadMNIST()

        self.setWindowTitle('Handwritten Digit Recogniser')
        self.setGeometry(300, 300, 300, 200)

        # Configure size of window
        self.resize(600, 400)
        self.show()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    ex = MyApp()
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
