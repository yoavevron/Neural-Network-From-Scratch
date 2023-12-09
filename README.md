# Basic Artificial Neural Network

This project established in order to reach a comprehensive understanding of the artificial neural network learning mechanism. An essential component to achieve this goal is to delve into the pure mathematical formulas implementation of a plain-vanilla basic neural network algorithm. This repository aims to serve as a valuable resource for those seeking to enhance their understanding by hands-on exploration of the foundations of that concept.
- Attempted to implement the simplest fully connected Multi Layer Perceptron
- Only usage of mathematics and numpy (without tensorflow/pytorch)
- Intended for hand-written digits image classification (MNIST Data Set)

## Quick Start
Clone repository:
```
git clone https://github.com/yoavevron/Basic-Artificial-Neural-Network.git
```
Run:
```
python main.py
```
### Adjusting network parameters (optional)
Inside the **main.py** adjust:
* The parameters that are given to neural_network.train_model
* The image to predict
* Note: when choosing evaluating you'll be asked to choose a model to evaluate

<br />Inside the **train_model** function in **neural_network.py** adjust:
* Initalized weights and biases values method (random/xavier/he)
* loss function (default: Categorical Cross Entropy)
* optimizers (default: Stochastic Gradient Descent)

<br />Inside **forward_propagation** function in **neural_network.py** adjust:
* Activation function of the hidden layers (default: ReLu)
### Changing train/test data (optional)
Data format:
* Both train and test data should be saved in CSVs in the proper folder as saved in the project
* File format inside CSV: (rows = images, columns=flatten vector of the images' pixels). For example, a CSV of 1000 images in size of 28X28 image will be organized that every row will represent another image, every column represent another pixel of the image (so 1000 rows x 784 columns). In addition, the first column represent the image real value. So in total the table will be (1000 X 785) 
* If the datasets files format is changed, it's crucial to update the load_data function inside neural_network.py to read the data properly
* The datasets organizing architecture is built so the data is saved in any number of (CSV) files to deal with the limit of maximum size of files that can be commited to github (50MB)
