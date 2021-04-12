#Image Processing
import numpy as np
import matplotlib.pyplot as plt

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
import torch.nn.functional as F
from torch.utils import data
import time


#AI PARAMETERS
epochNum = 10
batch_size = 64
learning_rate = 0.02
device = 'cuda' if cuda.is_available() else 'cpu'

def initAndLoadMNIST():
    datasetTransform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
    ])
    trainData = datasets.MNIST(root = 'Data\TrainData', train = True, transform = datasetTransform, download = True)
    testData = datasets.MNIST(root = 'Data\TestData', train = False, transform = datasetTransform)

    #Load data with transformations
    trainLoader = data.DataLoader(dataset = trainData, batch_size = batch_size, shuffle = True)
    testLoader = data.DataLoader(dataset = testData, batch_size = batch_size, shuffle = False)


    # Showing images

    # dataiter = iter(trainLoader)
    # images, labels = dataiter.next()

    # im2display = images[1].numpy().squeeze()

    # plt.imshow(im2display, interpolation='nearest', cmap='gray_r')
    # plt.show()


#GUI Related Content
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *



# This class will be a 2nd main window and will switch between the 2 upon event
class canvas(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.show()

        # Window configurations
        top = 400
        left = 400
        width = 600
        height = 400

        self.setWindowTitle("Drawing Recognition")
        self.setGeometry(top, left, width, height)

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
        
        # Save file in image format onto hard drive for analysis
        saveAction = QAction(QIcon("Icons\save.png"), "Save", self)
        saveAction.setShortcut("Ctrl+S")
        saveAction.triggered.connect(self.save)

        # Clear canvas
        clearAction = QAction(QIcon("Icons\clear.png"), "Clear", self)
        clearAction.setShortcut("Ctrl+C")
        clearAction.triggered.connect(self.clear)

        # Action to recognise the number drawn
        recogniseAction = QAction(QIcon("Icons\write.jpg"), "Recognise", self)
        recogniseAction.triggered.connect(self.clear)

        # Select Linear Model (Linear by default, so this dropdown is only for aesthetics)
        linearModel = QAction(QIcon("Icons\linearModel.png"), "Linear", self)
        linearModel.triggered.connect(self.clear)

        # Adding actions to drop down menus
        fileMenu.addAction(clearAction)
        fileMenu.addAction(saveAction)
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
            painter.setPen(QPen(Qt.black, 4, Qt.SolidLine,
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
    def save(self):
        filePath, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG(*.png);;JPEG(*.jpg *.jpeg);; ALL Files(*.*)")
        if filePath == "":
            return
        self.canvasImage.save(filePath)

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
        self.resize(600, 400)
        self.show()

    # closes parent window and opens child window by setting opacity to 0
    def callAnotherQMainWindow(self):
        win = canvas(self)
        self.setWindowOpacity(100) # Set to 0. if you want to toggle between windows, otherwise set to 100 if you want both open
 

if __name__ == '__main__':
    app = QApplication(sys.argv)

    ex = mainWindow()
    sys.exit(app.exec_())



#Note all methods below need to be adjusted for scope of variables and may need to be removed from methods



#Linear Model
class TestNet(nn.Module):
  def __init__(self):
    super(TestNet, self).__init__()
    self.l1 = nn.Linear(784, 700)
    self.l2 = nn.Linear(700, 350)
    self.l3 = nn.Linear(350, 175)
    self.l4 = nn.Linear(175, 85)
    self.l5 = nn.Linear(85, 30)
    self.l6 = nn.Linear(30, 15)
    self.l7 = nn.Linear(15, 10)

  def forward(self, x):
    x = x.view(-1, 784)
    x = F.relu(self.l1(x))
    x = F.relu(self.l2(x))
    x = F.relu(self.l3(x))
    x = F.relu(self.l4(x))
    x = F.relu(self.l5(x))
    x = F.relu(self.l6(x))
    x = self.l7(x)
    return F.log_softmax(x)

#

model = TestNet()
model.to(device)
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.5)

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

def TrainOverEpochs(epochNum, LOADER):
    final_accuracy = 0
    Training_Progress = 0
    account = 0
    flag = 0
    for epoch in range (1, epochNum + 1):
        #print(epoch)
      model.train()
      for i, (data, target) in enumerate(TRAINLOADER):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        Training_Progress = ((i/(len(TRAINLOADER)))/epochNum) * 100 + account
        if(flag == 0):
          if (Training_Progress %10 == 0):
            flag = 1
        elif(flag == 1):
          if (Training_Progress %10 == 0):
            account += 10
            Training_Progress += 10 
            #flag = 0
        #print (Training_Progress)
      if(epoch == epochNum):
        final_accuracy = testAccuracyModel(LOADER)
    #torch.save(model.state_dict(), 'C:/Users/krish/Desktop/KRISHEN AI FILES/SAVEDMODEL')
    return (final_accuracy)


#Basic Code for Probability Graph
#Need to Implement a way to get a list of probabilities

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