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

Digit Recogniser Version **1.XX**
    Changes:
        - 
        -
        -
        -
Digit Recogniser Version **1.XX**
    Changes:
        -
        -
        -
        -
        
