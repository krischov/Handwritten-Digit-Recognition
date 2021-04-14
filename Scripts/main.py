#Image Processing
import numpy as np
import matplotlib.pyplot as plt

#GUI Related Content
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import time

#AI Content
from torchvision.datasets import MNIST
import torch
from torchvision import datasets, transforms
import torchvision.models as models
from torch import nn, optim, cuda
import torch.nn.functional as F
from torch.utils import data


#AI PARAMETERS
epochNum = 1
batch_size = 64
learning_rate = 0.01
device = 'cuda' if cuda.is_available() else 'cpu'

#Global Variable
global Current_Training_Progress




#Linear Model
class TestNet(nn.Module):
  def __init__(self):
    super(TestNet, self).__init__()
    self.l1 = nn.Linear(784, 450)
    self.l2 = nn.Linear(450, 250)
    self.l3 = nn.Linear(250, 70)
    self.l4 = nn.Linear(70, 10)

  def forward(self, x):
    x = x.view(-1, 784)
    x = F.relu(self.l1(x))
    x = F.relu(self.l2(x))
    x = F.relu(self.l3(x))
    x = self.l4(x)
    return F.log_softmax(x)

# Neural network configuration and creation
model = TestNet()
model.to(device)
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.5)


# Tests for accuracy of the model and prints a percentage
def testAccuracyModel(LOADER):
    model.eval()
    Test_Correct = 0
    Loader_size = len(LOADER.dataset)
    for data, target in LOADER:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]
        Test_Correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    Accuracy = 100 * (Test_Correct/Loader_size)
    return Accuracy


#Basic Code for Probability Graph
def ShowProbabilityGraph(Loader):
    data, target = next(iter(Loader))
    img = data[0].view(1, 784)
    ConvertedLogValue = torch.exp(model(img))
    ProbabilityList = list(ConvertedLogValue.detach().numpy()[0])
    label = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    plt.barh(label,ProbabilityList)
    plt.title('Class Probability')
    plt.ylabel('Number')
    plt.xlabel('Probability')
    plt.show()


#Transforms
datasetTransform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5)),
])

#Transform for our own digit
datasetTransform2 = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5)),
])

# Train and test data
trainData = datasets.MNIST(root = 'Data\TrainData', train = True, transform = datasetTransform, download = True)
testData = datasets.MNIST(root = 'Data\TestData', train = False, transform = datasetTransform, download = True)

#Load data with transformations
trainLoader = data.DataLoader(dataset = trainData, batch_size = batch_size, shuffle = True)
testLoader = data.DataLoader(dataset = testData, batch_size = batch_size, shuffle = False)


# Reading image file
recognitionDataset = datasets.ImageFolder("Tests", transform = datasetTransform2)
Data = data.DataLoader(dataset = recognitionDataset, batch_size = 1, shuffle = False)




# TrainOverEpochs(epochNum, testLoader)

# model = torch.load('model')
# model.eval()


# Shows probability of which number the digit is likely to be
# ShowProbabilityGraph(Data)





################# GUI CONFIGURATIONS AND IMPLEMENTATIONS #################



# This class will be a 2nd main window and will switch between the 2 upon event
class canvas(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.show()

        # Window configurations
        top = 400
        left = 400
        width = 560
        height = 560

        self.setWindowTitle("Drawing Recognition")
        self.setFixedSize(width, height)

        # Configure drawing canvas and colour format to be grayscale. Also make canvas white
        self.canvasImage = QImage(self.size(), QImage.Format_Grayscale8)
        self.canvasImage.fill(Qt.white)
        

        # Drawing pen configurations
        self.drawing = False
        self.lastPoint = QPoint()

        # Menu bar setup
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu("File")
        modelMenu = mainMenu.addMenu("Model")
        
        # # Save file in image format onto hard drive for analysis
        # saveAction = QAction(QIcon("Icons\save.png"), "Save", self)
        # saveAction.setShortcut("Ctrl+S")
        # saveAction.triggered.connect(self.save)

        # Clear canvas
        clearAction = QAction(QIcon("Icons\clear.png"), "Clear", self)
        clearAction.setShortcut("Ctrl+C")
        clearAction.triggered.connect(self.clear)

        # Action to recognise the number drawn
        recogniseAction = QAction(QIcon("Icons\write.jpg"), "Recognise", self)
        recogniseAction.setShortcut("Ctrl+R")
        recogniseAction.triggered.connect(self.saveAndRecognise)

        # Select Linear Model (Linear by default, so this dropdown is only for aesthetics)
        linearModel = QAction(QIcon("Icons\linearModel.png"), "Linear", self)
        linearModel.triggered.connect(self.clear)

        # Adding actions to drop down menus
        fileMenu.addAction(clearAction)
        fileMenu.addAction(recogniseAction)
        modelMenu.addAction(linearModel)

    # Close parent window
    def closeEvent(self, QCloseEvent):
        self.parent.setWindowOpacity(1.)

    # Check for mouse press
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.lastPoint = event.pos()

    # Check mouse move
    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) & self.drawing:
            painter = QPainter(self.canvasImage)
            painter.setPen(QPen(Qt.black, 40, Qt.SolidLine,
            Qt.RoundCap, Qt.RoundJoin))

            painter.drawLine(self.lastPoint, event.pos())
            self.lastPoint = event.pos()
            self.update()

    # Check mouse release
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False
    

    # Paints stroke from when mouse is clicked and follows
    def paintEvent(self, event):
        canvasPainter = QPainter(self)
        canvasPainter.drawImage(self.rect(), self.canvasImage, self.canvasImage.rect())
        
    # Save function for recognition
    def saveAndRecognise(self):
        self.scaledImage = self.canvasImage.scaled(28, 28)

        self.scaledImage.save('Scripts\digitDrawn.png')
        print(self.scaledImage.shape())

    def clear(self):
        self.canvasImage.fill(Qt.white)
        self.update()



# This class manages the main window and all the drop downs to train model and view images
class mainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()
        

    # Popup window for train model view
    def trainModelDialog(self):
        trainButton = QPushButton('Train', self)
        cancelButton = QPushButton('Cancel', self)
        progressBar = QProgressBar(self)
        progressLabel = QLabel('Progress: ')

        

        # Creating text box to append download progress status
        msg = QTextBrowser()
        def downloadDataset(self):
            completed = 0

            while completed < 100:
                completed += 0.0001
                progressBar.setValue(completed)
            msg.append('Local MNIST Dataset used due to download error')

        # Reset progress bar back to 0
        def resetProgressBar(self):
            progressBar.setValue(0)


        # Nested functions to train dataset, also iterates the progress bar
        def trainDataset(self):
            
            
            # Trains the model over x amount of epochs (20 in this case)
            def TrainOverEpochs(epochNum, LOADER):

                final_accuracy = 0
                Percentage_Progress = 0
                account = 0
                flag = 0
                msg.append("Training...")
                for epoch in range (1, epochNum + 1):
                    model.train()
                    for i, (data, target) in enumerate(trainLoader):
                        #Code that trains the model
                        data, target = data.to(device), target.to(device)
                        optimizer.zero_grad()
                        output = model(data)
                        loss = criterion(output, target)
                        loss.backward()
                        optimizer.step()
                        

                        # Ensuring calculations are not done too frequently
                        if (i % 10 == 0):

                            #Code that keeps track of the training progress
                            Batch_Progress = (i/(len(trainLoader)))
                            Percentage_Progress = Batch_Progress * 100 

                            if(flag == 0):
                                if(Percentage_Progress == 0):
                                    flag = 1

                            elif(flag == 1):
                                if(Percentage_Progress == 0):
                                    account += (100/(epochNum))
                            Current_Training_Progress = round(Percentage_Progress*(1/epochNum) + account)
                            print(Current_Training_Progress)

                        # Append current status to progress bar
                        global progress
                        progress = int(Current_Training_Progress)
                        progressBar.setValue(progress)

                    #Code that Calculates Final Accuracy
                    if(epoch == epochNum):
                        progressBar.setValue(100)
                        accuracy = round(float((testAccuracyModel(LOADER))), 2)
                        finalAccuracy = round(float(accuracy), 2)
                        print(finalAccuracy)

                        msg.append(("Final Accuracy is: {}%".format(finalAccuracy)))

                        
                        # torch.save(model.state_dict(), 'model')

            TrainOverEpochs(epochNum, testLoader)
        
        # Buttons, Labels, and text browser to show progress
        downloadMNIST = QPushButton('Download MNIST', self)
        trainButton.clicked.connect(trainDataset)
        downloadMNIST.clicked.connect(downloadDataset)
        cancelButton.clicked.connect(resetProgressBar)
        
        
        # Dialog configuration and size
        widget = QDialog(self)
        widget.setFixedSize(560, 560)
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

        
        
    # Adds image and resizes to a dropdown menu
    def openImage(self):
        image = QPixmap('Icons\MNIST.PNG')
        pic = QLabel(self)
        pic.resize(600, 400)
        pic.setScaledContents(True)
        pic.setPixmap(image)
        pic.show()

    def initUI(self):
        viewTrainingImages = QAction('View Training Images', self)
        viewTrainingImages.triggered.connect(self.callAnotherQMainWindow)

        
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
        self.setFixedSize(560, 560)
        self.show()

    # closes parent window and opens child window by setting opacity to 0
    def callAnotherQMainWindow(self):
        win = canvas(self)
        self.setWindowOpacity(0.) # Set to 0. if you want to toggle between windows, otherwise set to 100 if you want both open
 

if __name__ == '__main__':
    app = QApplication(sys.argv)

    ex = mainWindow()
    sys.exit(app.exec_())