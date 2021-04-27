#Image Processing
import numpy as np
import matplotlib.pyplot as plt
import cv2

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
epochNum = 10
batch_size = 64
learning_rate = 0.07
device = 'cuda' if cuda.is_available() else 'cpu'


# Global Variables
global Current_Training_Progress
global PredictedNum

Current_Model_Index = 0
trainData = None
testData = None
trainLoader = None
testLoader = None

#Boolean that keeps track if MNIST dataset is downloaded
MNIST_DOWNLOADED = False

#Flag that checks if model 1 or model 2 is selected
#Initialised to zero originally
flag = 0

#Boolean that checks if a model has been trained since program start
M_Initialised = False

#Stores string of what model is trained
M_TRAINED = "NONE"

M_ACCURACY = 0

#Linear Model with 1 input layer, 3 hidden layers and 1 output layer
class TestNet(nn.Module):
  def __init__(self):
    super(TestNet, self).__init__()
    self.l1 = nn.Linear(784, 576)
    self.l2 = nn.Linear(576, 378)
    self.l3 = nn.Linear(378, 256)
    self.l4 = nn.Linear(256, 144)
    self.l5 = nn.Linear(144, 10)

  def forward(self, x):
      #Flattens data
    x = x.view(-1, 784)
    #Performs ReLU
    x = F.relu(self.l1(x))
    x = F.relu(self.l2(x))
    x = F.relu(self.l3(x))
    x = F.relu(self.l4(x))
    x = self.l5(x)
    #Log softmax so the probabilites of each class is obtained
    return F.log_softmax(x)
  
#Convolutional Model
class ConvNet(nn.Module):
  def __init__(self):
    super(ConvNet, self).__init__()
    #2 convolutional layers with Relu and pooling
    self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 28, kernel_size = 5, padding = 2)
    self.conv2 = nn.Conv2d(in_channels = 28, out_channels = 156,  kernel_size = 3, padding = 1)
    self.Pool = nn.MaxPool2d(2 , 2)
    self.l1 = nn.Linear(156*7*7, 156)
    #linear output layer
    self.l2 = nn.Linear(156 , 10)

  def forward(self, x):
    x = self.conv1(x)
    x = F.relu(x)
    x = self.Pool(x)

    x = self.conv2(x)
    x = F.relu(x)
    x = self.Pool(x)

    #Flattens data for the linear layer
    x = x.view(-1, 156*7*7)

    x = F.relu(self.l1(x))
    x = self.l2(x)
    #Log softmax so the probabilites of each class is obtained
    return F.log_softmax(x)

# Neural network configuration and creation

#Initialises models, sends those to the correct device, defines loss function and optimisers.
model1 = TestNet()
model1.to(device)
model2 = ConvNet()
model2.to(device)
criterion = nn.NLLLoss() #Loss function
optimizer1 = optim.SGD(model1.parameters(), lr = learning_rate, momentum = 0.5)
optimizer2 = optim.SGD(model2.parameters(), lr = learning_rate, momentum = 0.5)

#Methods for error messages
#Model and trained model arent the same
def model_MismatchMsg():
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setWindowTitle("Model Mismatch")
    msg.setText("Selected model and trained model are not the same.")
    x = msg.exec_()
    
#Model is not trained
def model_NotTrainedMsg():
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setWindowTitle("Error: Model not trained")
    msg.setText("Please train model.")
    x = msg.exec_()

#MNIST is not downloaded
def model_MNISTNotDownloadedMsg():
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setWindowTitle("Error")
    msg.setText("MNIST is not downloaded!")
    x = msg.exec_()

#Model is not selected
def model_NotSelectedMsg():
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setWindowTitle("Error")
    msg.setText("Model not selected!")
    x = msg.exec_()

# Tests for accuracy of the model and prints a percentage
#This function was heavily inspired from the lab documentation 
   #Source: https://colab.research.google.com/drive/1EGMH2qYQydshkOCXpciKhI2NXNJDfcyQ?usp=sharing
def testAccuracyModel(LOADER):
    #Check if in linear model mode
    if(flag == 1):
        model1.eval()
        Test_Correct = 0
        Loader_size = len(LOADER.dataset)
        for data, target in LOADER:
            data, target = data.to(device), target.to(device) #send to proper device
            output = model1(data)
            pred = output.data.max(1, keepdim=True)[1]
            Test_Correct += pred.eq(target.data.view_as(pred)).cpu().sum() #Check if prediciton was correct
        Accuracy = 100 * (Test_Correct/Loader_size)
        return Accuracy
    #Check if in convolutional model mode
    elif(flag == 2):
        model2.eval()
        Test_Correct = 0
        Loader_size = len(LOADER.dataset)
        for data, target in LOADER:
            data, target = data.to(device), target.to(device) #send to proper device
            output = model2(data) #get output
            pred = output.data.max(1, keepdim=True)[1]
            Test_Correct += pred.eq(target.data.view_as(pred)).cpu().sum() #Check if prediciton was correct
        Accuracy = 100 * (Test_Correct/Loader_size)
        return Accuracy       



# Probability graph when using Linear or Convolutional method
# The implementation of the exponents was inspired by: 
    #Source: https://towardsdatascience.com/handwritten-digit-mnist-pytorch-977b5338e627
    #Refer to step 6.
def ShowProbabilityGraph(Loader):
    global PredictedNum
    PredictedNum = 0
    if(flag == 1): 
        data, target = next(iter(Loader))
        data, target = data.to(device), target.to(device) #Send to correct device
        img = data[0].view(1, 784) #Flatten saved image
        ConvertedLogValue = torch.exp(model1(img)) #exponential to convert from logsoftmax to softmax
        ConvertedLogValue = ConvertedLogValue.cpu() 
        ProbabilityList = list(ConvertedLogValue.detach().numpy()[0]) #List of all class probabilites
        PredictedNum = ProbabilityList.index(max(ProbabilityList)) #Index with the highest probability number is the predicted num
        #Plot classes against probabilities
        label = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        plt.barh(label,ProbabilityList)
        plt.title('The Predicted Number is: %i'  %PredictedNum)
        plt.ylabel('Number')
        plt.xlabel('Probability')
        plt.show()

    elif(flag == 2):
        data, target = next(iter(Loader))
        data, target = data.to(device), target.to(device) #Send to correct device
        img = data[0].unsqueeze(0) #Get all dimensions for saved image
        ConvertedLogValue = torch.exp(model2(img)) #exponential to convert from logsoftmax to softmax
        ConvertedLogValue = ConvertedLogValue.cpu()
        ProbabilityList = list(ConvertedLogValue.detach().numpy()[0]) #List of all class probabilites
        PredictedNum = ProbabilityList.index(max(ProbabilityList)) #Index with the highest probability number is the predicted num
        #Plot classes against probabilities
        label = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        plt.barh(label,ProbabilityList)
        plt.title('The Predicted Number is: %i'  %PredictedNum)
        plt.ylabel('Number')
        plt.xlabel('Probability')
        plt.show()


# Calculate predicted number from drawn digit
# Similar to above, however updates a global variable to show what number was predicted
# Source: https://towardsdatascience.com/handwritten-digit-mnist-pytorch-977b5338e627
# Refer to step 6.
def probabilityCalc(Loader):
    global PredictedNum
    PredictedNum = 0
    #Linear Mode
    if(flag == 1): 
        data, target = next(iter(Loader))
        data, target = data.to(device), target.to(device)
        img = data[0].view(1, 784)
        ConvertedLogValue = torch.exp(model1(img))
        ConvertedLogValue = ConvertedLogValue.cpu()
        ProbabilityList = list(ConvertedLogValue.detach().numpy()[0])
        PredictedNum = ProbabilityList.index(max(ProbabilityList))
    #Convolutional Mode 
    elif(flag == 2):
        data, target = next(iter(Loader))
        data, target = data.to(device), target.to(device)
        img = data[0].unsqueeze(0)
        ConvertedLogValue = torch.exp(model2(img))
        ConvertedLogValue = ConvertedLogValue.cpu()
        ProbabilityList = list(ConvertedLogValue.detach().numpy()[0])
        PredictedNum = ProbabilityList.index(max(ProbabilityList))


#Transforms
datasetTransform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5)),
])

#Transform for our own digit
datasetTransform2 = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), #ensures image is 1-channel
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5)),
])

################# GUI CONFIGURATIONS AND IMPLEMENTATIONS #################




# This class manages the main window and all the drop downs to train model and view images
# The inspiration for this method of implementing a drawing canvas was using the 
        # source: https://www.mfitzp.com/tutorials/bitmap-graphics/
        # The exact implementation was not ideal which is why the layout, functions, and features
        # were switched up to closely match our project brief.
class mainWindow(QMainWindow):
    global PredictedNum
    def __init__(self):
        super().__init__()
        self.initUI()

        self.setWindowTitle("Drawing Recognition")
        self.setFixedSize(560, 560)

        # Configure drawing canvas and colour format to be grayscale. Also make canvas white
        self.canvasImage = QImage(self.size(), QImage.Format_Grayscale8)
        self.canvasImage.fill(Qt.white)
        self.canvasLabel = QLabel()
        self.pixmap = QPixmap.fromImage(self.canvasImage)
        self.canvasLabel.setPixmap(self.pixmap.scaled(560,560))
        

        # Write the most likely digit that was drawn to the screen
        self.probability = QTextEdit()
        self.probability.setReadOnly(True)
        self.probability.setFontPointSize(30)
        

        # Layouts to set canvas and buttons (canvas on left, buttons on right)
        self.vBox = QVBoxLayout()
        self.hBox = QHBoxLayout()
        self.formWidget = QWidget()
        self.recogniseButton = QPushButton('Recognise', self)
        self.recogniseButton.clicked.connect(self.saveAndRecognise)

        # Clears canvas
        self.clearButton = QPushButton('Clear', self)
        self.clearButton.clicked.connect(self.clear)

        # Displays graph if user wishes to see full probabilities of digit
        self.displayGraph = QPushButton('Full Graph', self)



        # Adding widgets and stretch to -1 to reduce gaps between buttons
        self.hBox.addWidget(self.canvasLabel)
        self.vBox.addWidget(self.recogniseButton)
        self.vBox.addWidget(self.clearButton)
        self.vBox.addWidget(self.probability)
        self.vBox.addWidget(self.displayGraph)
        self.vBox.addStretch(-1)
        self.hBox.addLayout(self.vBox)


        # Drawing pen configurations
        self.drawing = False
        self.lastPoint = QPoint()
    

    # Function which sets the layout and central widget later on, all else created earlier
    def createCanvas(self):
        self.formWidget.setLayout(self.hBox)
        self.setCentralWidget(self.formWidget)


        # Save function for recognition
    def saveAndRecognise(self):
        global PredictedNum
        

        # Uncomment if you want the output to be an inverted image
        self.canvasImage.invertPixels()
        self.scaledImage = self.canvasImage.scaled(28, 28)
        
        self.scaledImage.save('Tests/Numbers/digitDrawn.png')
        self.canvasImage.invertPixels()

        recognitionDataset = datasets.ImageFolder("Tests", transform = datasetTransform2)
        Data = data.DataLoader(dataset = recognitionDataset, batch_size = 1, shuffle = False)
        global Model_Mismatch

        def showGraph(self):
            ShowProbabilityGraph(Data)

        # Try and except to check if a model is loaded, if not, present error to train model
        if(MNIST_DOWNLOADED == True):
            if(M_Initialised == True):
                if(flag == 0):
                    model_NotSelectedMsg()
                #Linear Mode
                elif(flag == 1):
                    try:
                        # Display probability graph if button clicked, otherwise append most likely number
                        self.probability.clear()
                        loadModel = model1.load_state_dict(torch.load('model\model.pth', map_location = device))
                        probabilityCalc(Data)

                        self.probability.append(str(PredictedNum))
                        self.displayGraph.clicked.connect(showGraph)

                    except RuntimeError:
                        #Error message
                        model_MismatchMsg()
                #Convolutional Mode
                elif(flag == 2):
                    try:
                        # Display probability graph if button clicked, otherwise append most likely number
                        self.probability.clear()
                        loadModel = model2.load_state_dict(torch.load('model\model.pth', map_location = device))
                        probabilityCalc(Data)
                        self.probability.append(str(PredictedNum))
                        self.displayGraph.clicked.connect(showGraph)

                    except RuntimeError:
                        model_MismatchMsg() 
            elif(M_Initialised == False):
                if(flag == 0):
                    model_NotSelectedMsg()
                else:
                    model_NotTrainedMsg()
        elif(MNIST_DOWNLOADED == False):
            model_MNISTNotDownloadedMsg()

    # Function that clears the canvas, and also clears the predicted value from widget
    def clear(self):
        self.canvasImage.fill(Qt.white)
        self.update()
        self.probability.clear()


    # Check for mouse press, input is recognised as the left mouse button
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.lastPoint = event.pos()

    # Check mouse move
    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) & self.drawing:
            painter = QPainter(self.canvasImage)
            painter.setPen(QPen(Qt.black, 50, Qt.SolidLine,
            Qt.RoundCap, Qt.RoundJoin))

            painter.drawLine(self.lastPoint, event.pos())
            self.lastPoint = event.pos()
            self.update()

    # Check mouse release, input is recognised when left mouse button is released
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False
    

    # Paints stroke from when mouse is clicked and follows
    def paintEvent(self, event):
        canvasPainter = QPainter(self.canvasLabel.pixmap())
        canvasPainter.drawImage(self.rect(), self.canvasImage, self.canvasImage.rect())
        

    # Popup window for train model view
    def trainModelDialog(self):
        trainButton = QPushButton('Train', self)
        progressBar = QProgressBar(self)
        progressLabel = QLabel('Progress: ')
        PreloadButton = QPushButton('Use Saved', self)
        SaveButton = QPushButton('Save Model', self)

        # Creating text box to append download progress status
        msg = QTextBrowser()

        #Every time the dialog box is opened, check MNIST state
        if(MNIST_DOWNLOADED == True):
            MNIST_STATE = "YES"
        else: MNIST_STATE = "NO"

        #Every time the dialog box is opened, check ACCURACY value
        if(M_ACCURACY == 0):
            Accuracy = "UNDEFINED as no model is trained."
        else: Accuracy =  "{}%".format(M_ACCURACY)

        msg.append("MNIST Downloaded: " + MNIST_STATE)
        #Every time the dialog box is opened, check ACCURACY value
        msg.append("Trained Model is: " + M_TRAINED)

        msg.append("Model Accuracy is: " + Accuracy)
    
        # Function to download the MNIST dataset
        def downloadDataset(self):
            completed = 0

            #Attempt to download the train and test sets from mnist, then attempt to load the sets into the train and test loaders for the ai.
            #The train/test sets and loaders are updates globally so they can be accessed anywhere.
            #If this is performed successfully, update the Global MNIST downloaded flag.
            try:
                msg.append('Working...')
                global trainData
                trainData = datasets.MNIST(root = 'Data\TrainData', train = True, transform = datasetTransform, download = True)
                msg.append('Training Set Downloaded')
                progressBar.setValue(33)
                global testData
                testData = datasets.MNIST(root = 'Data\TestData', train = False, transform = datasetTransform, download = True)
                msg.append('Testing Set Downloaded')
                progressBar.setValue(66)
                global trainLoader
                trainLoader = data.DataLoader(dataset = trainData, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = 4)
                global testLoader
                testLoader = data.DataLoader(dataset = testData, batch_size = batch_size, shuffle = False)
                msg.append('MNIST Dataset Successfully Downloaded')
                global MNIST_DOWNLOADED
                MNIST_DOWNLOADED = True
            #If the download fails, do not crash the program and instead print out an error message.
            except: 
                msg.append('HTTP Error 503: Service Unavailable')
                msg.append('Please Locally Install MNIST Dataset into Data folder.')
            #At the end of the function, fill the progress bar as to indicate the function is completed.
            progressBar.setValue(100)


        # Nested functions to train dataset, also iterates the progress bar
        def trainDataset(self):
            if(MNIST_DOWNLOADED == True):

                # Trains the model over x amount of epochs (10 in this case)
                #The training function of the model was inspired by the lab work.
                    #Source: https://colab.research.google.com/drive/1EGMH2qYQydshkOCXpciKhI2NXNJDfcyQ?usp=sharing
                    # Refer to last block of code.
                def TrainOverEpochs(epochNum):
                    global progress
                    global M_Initialised
                    global M_TRAINED
                    global M_ACCURACY
                    final_accuracy = 0 
                    Percentage_Progress = 0
                    account = 0 #Adjustment amount for total progress
                    flag_local = 0 #Flag for progress calculaton
                    #No model selected
                    if(flag == 0):
                        msg.append("Model must be selected.")
                        progressBar.setValue(100)
                    #Linear Mode
                    elif(flag == 1):
                        msg.append("Training Linear Model...")
                        for epoch in range (1, epochNum + 1):
                            model1.train()
                            for i, (data, target) in enumerate(trainLoader):

                                #Code that trains the model
                                data, target = data.to(device), target.to(device) #send data to correct device
                                optimizer1.zero_grad() #Clear any calculated gradients
                                output = model1(data) #Get model output
                                loss = criterion(output, target) #Calculate loss
                                loss.backward() # get loss gradients
                                optimizer1.step() #Optimise weights

                                # Ensuring calculations are not done too frequently
                                if (i % 10 == 0):

                                    #Code that keeps track of the training progress
                                    Batch_Progress = (i/(len(trainLoader))) 
                                    Percentage_Progress = Batch_Progress * 100 #Percentage of batch completed
                                    if(flag_local == 0):
                                        if(Percentage_Progress == 0):
                                            flag_local = 1
                                    elif(flag_local == 1):
                                        if(Percentage_Progress == 0):
                                            account += (100/(epochNum))
                                    Current_Training_Progress = round(Percentage_Progress*(1/epochNum) + account) #True progress calculation
                                    
                                # Append current status to progress bar
                                progress = int(Current_Training_Progress)
                                progressBar.setValue(progress)

                        #Code that Calculates Final Accuracy
                        if(epoch == epochNum):
                            progressBar.setValue(100)
                            accuracy = round(float((testAccuracyModel(testLoader))), 2)
                            finalAccuracy = round(float(accuracy), 2)
                            msg.append("Linear Model Trained")
                            M_TRAINED = "Linear"
                            msg.append(("Final Accuracy is: {}%".format(finalAccuracy)))
                            M_ACCURACY = finalAccuracy

                            # Saves model so you don't need to retrain
                            torch.save(model1.state_dict(), 'model\model.pth')
                            M_Initialised = True
                    #Convolutional Mode
                    #Note commenting is very similar to above (due to identical nature)
                    #Commenting removed to reduce clutter
                    elif(flag == 2):
                        msg.append("Training Convolutional Model...")
                        for epoch in range (1, epochNum + 1):
                            model2.train()
                            for i, (data, target) in enumerate(trainLoader):

                                #Code that trains the model
                                data, target = data.to(device), target.to(device)
                                optimizer2.zero_grad()
                                output = model2(data)
                                loss = criterion(output, target)
                                loss.backward()
                                optimizer2.step()

                                # Ensuring calculations are not done too frequently
                                if (i % 10 == 0):

                                    #Code that keeps track of the training progress
                                    Batch_Progress = (i/(len(trainLoader)))
                                    Percentage_Progress = Batch_Progress * 100
                                    if(flag_local == 0):
                                        if(Percentage_Progress == 0):
                                            flag_local = 1
                                    elif(flag_local == 1):
                                        if(Percentage_Progress == 0):
                                            account += (100/(epochNum))
                                    Current_Training_Progress = round(Percentage_Progress*(1/epochNum) + account)

                                # Append current status to progress bar
                                progress = int(Current_Training_Progress)
                                progressBar.setValue(progress)

                        #Code that Calculates Final Accuracy
                        if(epoch == epochNum):
                            progressBar.setValue(100)
                            accuracy = round(float((testAccuracyModel(testLoader))), 2)
                            finalAccuracy = round(float(accuracy), 2)
                            msg.append("Convolutional Model Trained")
                            M_TRAINED = "Convolutional"
                            msg.append(("Final Accuracy is: {}%".format(finalAccuracy)))
                            M_ACCURACY = finalAccuracy

                            # Saves model so you don't need to retrain
                            torch.save(model2.state_dict(), 'model\model.pth')
                            M_Initialised = True

                TrainOverEpochs(epochNum)
            elif(MNIST_DOWNLOADED == False):
                model_MNISTNotDownloadedMsg()
        
        # Function to use a preloaded model that is saved in the respective folder
        # If the model does not exist, return QDialog error saying it does not exist.
        def usePreloadedModel(self):
            global M_ACCURACY
            global M_TRAINED
            global M_Initialised
            if(MNIST_DOWNLOADED == True):
                #Error message code for when model is not selected
                if(flag == 0):
                    msg3 = QMessageBox()
                    msg3.setIcon(QMessageBox.Critical)
                    msg3.setWindowTitle("Model Unselected")
                    msg3.setText("Please select the model type you wish to load.")
                    x = msg3.exec_()
                    #Linear Mode
                elif(flag == 1):
                    try:
                        #Check if the right mode is selected by loading the model
                        loadModel = model1.load_state_dict(torch.load('Saved Model\Linear\model.pth', map_location = device))
                        msg.append("Working...")
                        progressBar.setValue(0)
                        #Obtain accuracy of loaded model
                        accuracy = round(float((testAccuracyModel(testLoader))), 2)
                        finalAccuracy = round(float(accuracy), 2)
                        #Update global variables
                        M_ACCURACY = finalAccuracy
                        M_TRAINED = "Linear"
                        msg.append("Pretrained Linear Model is loaded.")
                        msg.append(("Model Accuracy is: {}%".format(finalAccuracy)))
                        #Finally save the model to the 'Active' folder
                        torch.save(model1.state_dict(), 'model\model.pth')
                        M_Initialised = True
                        progressBar.setValue(100)
                    except RuntimeError:
                        #Error message for when trying to load a model in the wrong model mode
                        msg1 = QMessageBox()
                        msg1.setIcon(QMessageBox.Critical)
                        msg1.setWindowTitle("Error")
                        msg1.setText("Model does not match selected model")
                        x = msg1.exec_()
                    except FileNotFoundError:
                        #Error message for when no file is found in linear folder 
                        msg1 = QMessageBox()
                        msg1.setIcon(QMessageBox.Critical)
                        msg1.setWindowTitle("Error")
                        msg1.setText("There is no model in folder.")
                        x = msg1.exec_()                        
                elif(flag == 2):
                    try:
                        #Check if the right mode is selected by loading the model
                        loadModel = model2.load_state_dict(torch.load('Saved Model\Convolutional\model.pth', map_location = device))
                        msg.append("Working...")
                        progressBar.setValue(0)
                        #Obtain accuracy of loaded model
                        M_ACCURACY = testAccuracyModel(testLoader)
                        accuracy = round(float((testAccuracyModel(testLoader))), 2)
                        finalAccuracy = round(float(accuracy), 2)
                        #Update global variables
                        M_ACCURACY = finalAccuracy
                        M_TRAINED = "Convolutional"
                        msg.append("Pretrained Convolutional Model is loaded.")
                        msg.append(("Model Accuracy is: {}%".format(finalAccuracy)))
                        #Finally save the model to the 'Active' folder
                        torch.save(model2.state_dict(), 'model\model.pth')
                        M_Initialised = True
                        progressBar.setValue(100)
                    except RuntimeError:
                        #Error message for when trying to load a model in the wrong model mode
                        msg1 = QMessageBox()
                        msg1.setIcon(QMessageBox.Critical)
                        msg1.setWindowTitle("Error")
                        msg1.setText("Model does not match selected model")
                        x = msg1.exec_()
                    except FileNotFoundError:
                        #Error message for when no file is found in linear folder 
                        msg1 = QMessageBox()
                        msg1.setIcon(QMessageBox.Critical)
                        msg1.setWindowTitle("Error")
                        msg1.setText("There is no model in folder.")
                        x = msg1.exec_()                                
            elif(MNIST_DOWNLOADED == False):
                model_MNISTNotDownloadedMsg()

        def SaveModel(self):
            if(MNIST_DOWNLOADED == True):
                if(flag == 0):
                    #Error message if no model is selected
                    msg3 = QMessageBox()
                    msg3.setIcon(QMessageBox.Critical)
                    msg3.setWindowTitle("Model Unselected")
                    msg3.setText("Please select the model type you wish to save.")
                    x = msg3.exec_()
                    #Linear model
                elif(flag == 1):
                    try:
                        #First check to see if the model is the right type by loading it as a linear model
                        loadModel = model1.load_state_dict(torch.load('model\model.pth', map_location = device))
                        #If this passes, save it.
                        torch.save(model1.state_dict(), 'Saved Model\Linear\model.pth')
                        msg.append("Linear Model has been saved.")
                    except (RuntimeError):
                        #If model is of wrong type, display this image
                        msg1 = QMessageBox()
                        msg1.setIcon(QMessageBox.Critical)
                        msg1.setWindowTitle("Error")
                        msg1.setText("Model type in 'model' folder does not match selected model")
                        x = msg1.exec_()
                    except FileNotFoundError:
                        #Error message if there no 'active' model
                        msg1 = QMessageBox()
                        msg1.setIcon(QMessageBox.Critical)
                        msg1.setWindowTitle("Error")
                        msg1.setText("There is no model to be saved")
                        x = msg1.exec_()
                #Convolutional Mode
                elif(flag == 2):
                    try:
                        #First check to see if the model is the right type by loading it as a convolutional model
                        loadModel = model2.load_state_dict(torch.load('model\model.pth', map_location = device))
                        #If this passes, save it.
                        torch.save(model2.state_dict(), 'Saved Model\Convolutional\model.pth')
                        msg.append("Convolutional Model has been saved.")
                    except RuntimeError:
                        #If model is of wrong type, display this image
                        msg1 = QMessageBox()
                        msg1.setIcon(QMessageBox.Critical)
                        msg1.setWindowTitle("Error")
                        msg1.setText("Model type in 'model' folder does not match selected model")
                        x = msg1.exec_()
                    except FileNotFoundError:
                        #Error message if there no 'active' model
                        msg1 = QMessageBox()
                        msg1.setIcon(QMessageBox.Critical)
                        msg1.setWindowTitle("Error")
                        msg1.setText("There is no model to be saved")
                        x = msg1.exec_()              
            elif(MNIST_DOWNLOADED == False):
                model_MNISTNotDownloadedMsg()



        # Buttons, Labels, and text browser to show progress
        downloadMNIST = QPushButton('Download MNIST', self)
        trainButton.clicked.connect(trainDataset)
        downloadMNIST.clicked.connect(downloadDataset)
        PreloadButton.clicked.connect(usePreloadedModel)
        SaveButton.clicked.connect(SaveModel)

        # Changes to linear model mode
        def changeToLinearModel(self):
            global flag
            flag = 1
            msg.append(("Switched to Linear Model."))

        # Changes to convolutional model mode
        def changeToConvModel(self):
            global flag
            flag = 2
            msg.append(("Switched to Convolutional Model."))
            msg.append("Convolutional Model: Do not attempt to train without a CUDA device.")

        

        
        # Dialog configuration and size
        widget = QDialog(self)
        widget.resize(560, 600)
        widget.setWindowTitle('Dialog')
        widgetGrid = QGridLayout()

        # Function which activates model depending on option chosen
        #Global flag value 
        def onActivated(modelIndex):
            global Current_Model_Index
            Current_Model_Index = modelIndex
            if(modelIndex == 0):
                global flag
                flag = 0
                msg.append("Model must be selected.")
            if (modelIndex == 1):
                changeToLinearModel(self)
            if (modelIndex == 2):
                changeToConvModel(self)

        # Defining drop down box for models
        listOfModels = QComboBox(widget)
        listOfModels.addItem('Select Model')
        listOfModels.addItem('Linear')
        listOfModels.addItem('Convolutional')
        listOfModels.setCurrentIndex(Current_Model_Index)

        # Assigning index to variable which will be passed into the 
        # onActivated function for index change
        choice = listOfModels.currentText()

        # Checks for index change, if changed, calls onActivated function
        # with 'choice' as input
        listOfModels.currentIndexChanged.connect(onActivated)
        

    
        # Grid layout for buttons
        widget.setLayout(widgetGrid)
        widgetGrid.addWidget(msg, 0, 0, 1, 4)
        widgetGrid.addWidget(progressLabel, 1, 0)
        widgetGrid.addWidget(progressBar, 1, 1, 1, 4)
        widgetGrid.addWidget(downloadMNIST, 2, 0)
        widgetGrid.addWidget(trainButton, 2, 1)
        widgetGrid.addWidget(PreloadButton, 2, 2)
        widgetGrid.addWidget(SaveButton, 2, 3)
        widgetGrid.addWidget(listOfModels, 2, 4)
        

        widget.exec_()

        
        
     # Adds testing images into sub plot
    def openTestImages(self):
        if(MNIST_DOWNLOADED == True):
            # Global index to be used for next and previous batches
            global imageIndex
            imageIndex = 0

            # Main window initialisation
            testDialog = QMainWindow(self)
            testDialog.resize(560, 600)
            testDialog.setWindowTitle('MNIST Test Images')


            wid = QWidget(self)
            testDialog.setCentralWidget(wid)
            testDialogLayout = QGridLayout(wid)
            
            nextButton = QPushButton('Next Page', self)
            prevButton = QPushButton('Previous Page', self)
            
            

            # Initial creation of the first page of test values
            for i in range(1, 7):
                
                for j in range(1, 7):
                    image = testData[imageIndex][0]
                    
                    # Converts raw tensor to an ndarray and removes unecessary index
                    imageToDisplay = image.numpy().squeeze(0)
                    
                    # Convert floats generated in the ndarray to ints
                    imageToDisplay *= 255
                    imageToDisplay = imageToDisplay.astype(np.uint8)
                    
                    # Save image to hard disk and create a pixmap to store into a label on a grid
                    cv2.imwrite('image.png', imageToDisplay)
                    height, width = imageToDisplay.shape
                    newImage = QImage('image.png')
                    pixmap = QPixmap.fromImage(newImage)
                    pixmapImage = QPixmap(pixmap)
                    

                    label = QLabel()
                    pixmapImage = pixmapImage.scaled(56, 56)
                    label.setPixmap(pixmapImage)

                    
                    wid.setLayout(testDialogLayout)
                    testDialogLayout.addWidget(label, i, j)
                    imageIndex = imageIndex + 1


            # Function to go to next 'page' of test values
            def nextBatch(self):
                
                global imageIndex
                for i in range(1, 7):
                    for j in range(1, 7):
                        image = testData[imageIndex + 36][0]

                        # Converts raw tensor to an ndarray and removes unecessary index
                        imageToDisplay = image.numpy().squeeze(0)
                        
                        # Convert floats generated in the ndarray to ints
                        imageToDisplay *= 255
                        imageToDisplay = imageToDisplay.astype(np.uint8)
                        
                        # Save image to hard disk and create a pixmap to store into a label on a grid
                        cv2.imwrite('image.png', imageToDisplay)
                        height, width = imageToDisplay.shape
                        newImage = QImage('image.png')
                        pixmap = QPixmap.fromImage(newImage)
                        pixmapImage = QPixmap(pixmap)
                        

                        label = QLabel()
                        pixmapImage = pixmapImage.scaled(56, 56)
                        label.setPixmap(pixmapImage)

                        
                        wid.setLayout(testDialogLayout)
                        testDialogLayout.addWidget(label, i, j)

                        label = QLabel()
                        label.setPixmap(pixmapImage)

                        
                        wid.setLayout(testDialogLayout)
                        testDialogLayout.addWidget(label, i, j)
                        imageIndex = imageIndex + 1


            # Function to go to previous 'page' of test values
            def prevBatch(self):
                global imageIndex
                for i in range(1, 7):
                    for j in range(1, 7):
                        image = testData[imageIndex - 36][0]
                        
                        # Converts raw tensor to an ndarray and removes unecessary index
                        imageToDisplay = image.numpy().squeeze(0)
                        
                        # Convert floats generated in the ndarray to ints
                        imageToDisplay *= 255
                        imageToDisplay = imageToDisplay.astype(np.uint8)
                        
                        # Save image to hard disk and create a pixmap to store into a label on a grid
                        cv2.imwrite('image.png', imageToDisplay)
                        height, width = imageToDisplay.shape
                        newImage = QImage('image.png')
                        pixmap = QPixmap.fromImage(newImage)
                        pixmapImage = QPixmap(pixmap)
                        

                        label = QLabel()
                        pixmapImage = pixmapImage.scaled(56, 56)
                        label.setPixmap(pixmapImage)

                        
                        wid.setLayout(testDialogLayout)
                        testDialogLayout.addWidget(label, i, j)

                        label = QLabel()
                        label.setPixmap(pixmapImage)

                        # Reducing the index by 1 and setting layout
                        wid.setLayout(testDialogLayout)
                        testDialogLayout.addWidget(label, i, j)
                        imageIndex = imageIndex - 1

            nextButton.clicked.connect(nextBatch)
            prevButton.clicked.connect(prevBatch)

            testDialogLayout.addWidget(prevButton, 8, 6)
            testDialogLayout.addWidget(nextButton, 7, 6)
            testDialog.show()
        elif(MNIST_DOWNLOADED == False):
            model_MNISTNotDownloadedMsg()

    # Displays training images on a plot       
    def openTrainedImages(self):
        if(MNIST_DOWNLOADED == True):
            # Global index to be used for next and previous batches
            global imageIndex
            imageIndex = 0

            # Main window initialisation
            trainDialog = QMainWindow(self)
            trainDialog.resize(560, 600)
            trainDialog.setWindowTitle('MNIST Train Images')


            wid = QWidget(self)
            trainDialog.setCentralWidget(wid)
            trainDialogLayout = QGridLayout(wid)
            
            nextButton = QPushButton('Next Page', self)
            prevButton = QPushButton('Previous Page', self)
            
            

            # Initial creation of the first page of test values
            for i in range(1, 7):
                
                for j in range(1, 7):
                    image = trainData[imageIndex][0]
                    
                    # Converts raw tensor to an ndarray and removes unecessary index
                    imageToDisplay = image.numpy().squeeze(0)
                    
                    # Convert floats generated in the ndarray to ints
                    imageToDisplay *= 255
                    imageToDisplay = imageToDisplay.astype(np.uint8)
                    
                    # Save image to hard disk and create a pixmap to store into a label on a grid
                    cv2.imwrite('image.png', imageToDisplay)
                    height, width = imageToDisplay.shape
                    newImage = QImage('image.png')
                    pixmap = QPixmap.fromImage(newImage)
                    pixmapImage = QPixmap(pixmap)
                    

                    label = QLabel()
                    pixmapImage = pixmapImage.scaled(56, 56)
                    label.setPixmap(pixmapImage)

                    
                    wid.setLayout(trainDialogLayout)
                    trainDialogLayout.addWidget(label, i, j)
                    imageIndex = imageIndex + 1



            # Function to go to next 'page' of test values
            def nextBatch(self):
                global imageIndex
                for i in range(1, 7):
                    for j in range(1, 7):
                        image = trainData[imageIndex + 36][0]

                        # Converts raw tensor to an ndarray and removes unecessary index
                        imageToDisplay = image.numpy().squeeze(0)
                        
                        # Convert floats generated in the ndarray to ints
                        imageToDisplay *= 255
                        imageToDisplay = imageToDisplay.astype(np.uint8)
                        
                        # Save image to hard disk and create a pixmap to store into a label on a grid
                        cv2.imwrite('image.png', imageToDisplay)
                        height, width = imageToDisplay.shape
                        newImage = QImage('image.png')
                        pixmap = QPixmap.fromImage(newImage)
                        pixmapImage = QPixmap(pixmap)
                        

                        label = QLabel()
                        pixmapImage = pixmapImage.scaled(56, 56)
                        label.setPixmap(pixmapImage)

                        
                        wid.setLayout(trainDialogLayout)
                        trainDialogLayout.addWidget(label, i, j)

                        label = QLabel()
                        label.setPixmap(pixmapImage)

                        
                        wid.setLayout(trainDialogLayout)
                        trainDialogLayout.addWidget(label, i, j)
                        imageIndex = imageIndex + 1


            # Function to go to previous 'page' of test values
            def prevBatch(self):
                global imageIndex
                for i in range(1, 7):
                    for j in range(1, 7):
                        image = testData[imageIndex - 36][0]
                        
                        # Converts raw tensor to an ndarray and removes unecessary index
                        imageToDisplay = image.numpy().squeeze(0)
                        
                        # Convert floats generated in the ndarray to ints
                        imageToDisplay *= 255
                        imageToDisplay = imageToDisplay.astype(np.uint8)
                        
                        # Save image to hard disk and create a pixmap to store into a label on a grid
                        cv2.imwrite('image.png', imageToDisplay)
                        height, width = imageToDisplay.shape
                        newImage = QImage('image.png')
                        pixmap = QPixmap.fromImage(newImage)
                        pixmapImage = QPixmap(pixmap)
                        

                        label = QLabel()
                        label.setPixmap(pixmapImage)

                        
                        wid.setLayout(trainDialogLayout)
                        trainDialogLayout.addWidget(label, i, j)

                        label = QLabel()
                        pixmapImage = pixmapImage.scaled(56, 56)
                        label.setPixmap(pixmapImage)

                        # Reducing the index by 1 and setting layout
                        wid.setLayout(trainDialogLayout)
                        trainDialogLayout.addWidget(label, i, j)
                        imageIndex = imageIndex - 1

            nextButton.clicked.connect(nextBatch)
            prevButton.clicked.connect(prevBatch)

            trainDialogLayout.addWidget(prevButton, 8, 6)
            trainDialogLayout.addWidget(nextButton, 7, 6)
            trainDialog.show()
        elif(MNIST_DOWNLOADED == False):
            model_MNISTNotDownloadedMsg()

    # Initialises UI for the main window
    def initUI(self):
        viewTrainingImages = QAction('View Training Images', self)
        viewTrainingImages.triggered.connect(self.openTrainedImages)
        
        
        drawingCanvas = QAction('Drawing Canvas', self)
        drawingCanvas.triggered.connect(self.createCanvas)

        
        viewTestingImages = QAction('View Testing Images', self)

        viewTestingImages.triggered.connect(self.openTestImages)
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
        viewMenu = menubar.addMenu('&View')
        viewMenu.addAction(viewTrainingImages)
        viewMenu.addAction(viewTestingImages)
        viewMenu.addAction(drawingCanvas)

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