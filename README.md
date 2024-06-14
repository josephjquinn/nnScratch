# Fully Connected Neural Network using Numpy & Calculus 

This project was a learning project for me to better understand the fundamentals of Neural Networks and their architecture. I created this class without using any machine learning libraries doing most of the calculations by hand. 
This repository contains a simple neural network implementation in Python, designed for educational purposes. The neural network supports customizable architecture and training options.
## Introduction

Hand Gesture Recognition is a project designed to enable real-time interpretation of hand gestures using computer vision and machine learning techniques. By harnessing the capabilities of the Mediapipe library for hand tracking and the Random Forest classifier for gesture classification, this project offers a simple yet effective solution for recognizing and understanding hand gestures.

![Figure_1](https://github.com/josephjquinn/asl-model/assets/81782398/651c56d5-bbc7-49d0-971b-fa75aba3a667)
![Figure_2](https://github.com/josephjquinn/asl-model/assets/81782398/9f25fc88-c3d2-4b69-933c-18239dc2dae2)
![Figure_3](https://github.com/josephjquinn/asl-model/assets/81782398/86428435-dec7-4268-a8ae-bc672fabcc3a)


### Features

- Activation Functions: Supports ReLU, Sigmoid, and Leaky ReLU activation functions.
- Loss Function: Cross-entropy loss function for classification tasks.
- Initialization Methods: Supports initialization methods like random, He initialization, and normalized initialization.
- Training: Includes options for mini-batch training, with customizable batch size and learning rate.
- Monitoring: Real-time plots of training loss, validation loss, and accuracy during training.
- Prediction: Single and grid predictions on unseen data.
- Hyperparameter Optimization Script: Optimize and plot hyperparameters such as hidden layer neurons, learning rate, activation function, number of epochs, initialization method, and batch size.

### Design Notes

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/josephjquinn/nnScratch
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Usage

- `nn.py` Contains network class 
- `data.py` Contains data loader class 
- `loss.py` Contains loss functions 
- `activation.py` Contains activation functions 
- `main.py` trains model on MNIST dataset, contains grid prediction output  
- `hyper.py` calculates and saves optimal hyperparameters 

