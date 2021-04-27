-- How to install and run this project --

Library Installations:

To install the libraries, first download Anaconda (miniconda3) and run these commands in the CMD
You will also need Python 3.8.5/3.8.8 installed along with an editor to run the code.
- pip install opencv-python
- pip install torchvision
- pip install matplotlib
- pip install numpy
- pip install pyqt5


Running the Application:
- Open Folder - project-1-team_34
- Double click the **main.py** script in the **Scripts** folder
- Click Run and Debug on the left panel and select your Python Interpreter.
- The program should now open and can be used.
    * Ensure that the **whole directory** of the contents is **opened** on your editor

Training a model:
- To train a model, select File -> Train Model View
- Note that you must **download MNIST** using the button provided before proceeding.
- Download MNIST, then select a model from the dropdown between Linear and Convolutional.
    * Only use convolutional if you have CUDA installed.


Saving/Using a saved model
- Train the model, this may take a few minutes and do not click or use anything else while it trains.
- If you would like to load a saved model, copy your *model.pth* into the **Saved Model** folder in the directory 
    OR train a model and click **Save Model**
- Click **Use Saved** to use the saved model.


Drawing Digits:
- To begin recognising your drawn digits, go to the main window and click View -> Drawing Canvas.
    * This will open a sub-window which can be drawn on using your mouse. Click and hold the left mouse button to draw.
- Once the digit is drawn, click **File -> Recognise** or **CTRL + R** to bring up the probability graph.
- A new window will open displaying the likely probabilities of what the model thinks your digit is.
    * Once you are finished with your drawn digit, close the probability window and clear the canvas before drawing again.
- To clear the canvas, click **File -> Clear** or **CTRL + C**.

Viewing MNIST Testing/Training Images:
- To view the MNIST Testing or Training Images, go to the main window and click **View -> View Training Images/View Testing Images**
- A new window will open that will display a grid of 36 images from the respective option.
- The images can be traversed using the **Next Page** and **Previous Page** buttons.










-- Program iterations and updates --

Format:

Digit Recogniser Version **1.01**
    Changes:
        - Method added to clear the recognised number.


Digit Recogniser Version **1.00**
    Changes:
        - Canvas is now accesible directly from the main window.
        - Predicted Number that was drawn will display in a read only fashion.
        - Entire class for drawing canvas was removed due to the integration with the main window - 100's of code lines removed.
        - Button to show entire probability graph for user added. Opens up new window.
  
  
 Digit Recogniser Version **0.24**
    Changes:
        - Images are now scaled when viewing the MNIST images.


  Digit Recogniser Version **0.23**
    Changes:
        - Instead of being able to load and save only one model, the program can now save one convolutional model and one linear model respectively.
        - Higher accuracy pretrained model added to convolutional model folder.
        - Models can now be loaded irrespective of the device.
        - Added messages for loading and saving.
    
    
  Digit Recogniser Version **0.22**
    Changes:
        - Implemented a function that allows the user to save a model.
        - Implemented a function that allows the user to load a model.
        - Added high accuracy linear pretrained model.
        - Exceptions added to the functions so crashes do not occur. For example, attempting to load a model of the wrong type.


  Digit Recogniser Version **0.21**
    Changes:
        -  Removed cancel button from training window as it served no useful functionality.
        -  Added variables to keep track of MNIST downloaded state, what model is trained, and what that model accuracy was.
        -  Opening the training window will now show if MNIST is downloaded, what model is trained and what accuracy the model has.
        -  Turned error messages into methods to shorten code.
        -  Fixed event where two error messages would pop up.


  Digit Recogniser Version **0.20**
    Changes:
        - Addded error cases to prevent program crashing if MNIST was not downloaded
        - Added more error cases.


  Digit Recogniser Version **0.19**
    Changes:
        - Added a DownloadMNIST button and method in the training menu.


  Digit Recogniser Version **0.18**
    Changes:
        - The entire training and test dataset can now be viewed instead of only one batch. 
        - This includes all 60000 training images and 10000 testing images.
        - Added a page system where 36 images are viewed at a time.


  Digit Recogniser Version **0.17**
    Changes:
        - Added warning to when trying to train convolutional without CUDA due to time.
        - Increased linear model accuracy.
        - Model state is now remembered when reopening the training dialog i.e. if convolutional is selected it will be remembered.


  Digit Recogniser Version **0.16**
    Changes:
        - Dropdown combobox menu is now added. This allows for model switching between convolutional and linear.
        - Edited flag to be zero if no model is selected (i.e. a model should be selected) - (3  model states)


  Digit Recogniser Version **0.15**
    Changes:
        - Testing and training images can now be viewed (Only one batch).


  Digit Recogniser Version **0.14**
    Changes:
        - Probability graph implemented, allows to see the probability of number classes based on the model input.
        - Linear model can now view canvas image.


  Digit Recogniser Version **0.13**
    Changes:
        - Workers added to train loader to speed it up.
        - Convolutional model added.
        - Model is now saved at the end of training.


  Digit Recogniser Version **0.12**
    Changes:
        - Added progressbar to work with the training function. Can now see state of training progress.
        - Resolved error for training function - when using CUDA as a device.


  Digit Recogniser Version **0.11**
    Changes:
        - Error downloading MNIST from source, so local files are used.
        - Simplified TrainOverEpochs.
   

  Digit Recogniser Version **0.10**
    Changes:
        - Added scaling functionality to convert from our drawing size to a 28,28 size for loading and recognition.
        - Adjusted brush size so the output images are more representative.
        - Adding output files of numbers 0-9.


  Digit Recogniser Version **0.09**
    Changes:
        - Added functionality for painting on the window using the mouse.
        - Actions were added to the 'Save' and 'Clear'


  Digit Recogniser Version **0.08**
    Changes:
        - Implementation of ShowProbabilityGraph()
        - Takes in a loader. Converts the NN output into a list of probabilities. Constructs a Horizontal Probability Chart.


  Digit Recogniser Version **0.07**
    Changes:
        - Added simple probability graph code.
        - Accounted for when number reaches the next ten. For example 19 -> 20, but training progress would display the incorrect number.
        - Removed trainModel and merged with TrainOverEpochs()
        - Implemented code to keep track of the percentage completion of training progress.


  Digit Recogniser Version **0.06**
    Changes:
        - Added testModel base function, will be used to calculate our model accuracy.
        - Implemented TrainOverEpochs() which will train the model over epochNum epochs.
        - Will return the final model accuracy after training and will save the model.


  Digit Recogniser Version **0.05**
    Changes:
        - Created skeleton code for training a model.
        - Set model, set model device, set loss function and optimizer.
        - Filled in Model Parameters.


  Digit Recogniser Version **0.04**
    Changes:
        - Added a basic NN. Has 1 input and output layer, 5 hidden layers and uses LOGSOFTMAX for probability chart.
        - Added View Testing Images function into GUI.
       
       
  Digit Recogniser Version **0.03**
    Changes:
        - Completed Train Model View dropdown GUI.
        - Added time, torch utils, and cuda libraries.
        - Inverted image on plot using scikit-image library
        - Added grayscale transform to make the image 3 channels.


  Digit Recogniser Version **0.02**
    Changes:
        - Uploaded MNIST dataset.
        - Added download folders.
        - Added section for AI parameters.
        - Added function that will setup the TEST and TRAIN datasets. These are then loaded into DATALOADER.
        - Added a basic function for defining our model.
        - Added skeleton code for future methods.


  Digit Recogniser Version **0.01**
    Changes:
        - Created Main.
        - Created Basic GUI.
   
       
