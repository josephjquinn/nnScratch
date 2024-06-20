# Fully Connected Neural Network using Numpy & Calculus

This project was a learning project for me to better understand the fundamentals of Neural Networks and their architecture. I created this class without using any machine learning libraries doing most of the calculations by hand.
This repository contains a simple neural network implementation in Python, designed for educational purposes. The neural network supports customizable architecture and training options.

<img width="1400" alt="image" src="https://github.com/josephjquinn/nnScratch/assets/81782398/6d4ea186-1baf-44cf-96c3-cfda0a8138dc">
<img width="400" alt="image" src="https://github.com/josephjquinn/nnScratch/assets/81782398/bac57e3d-94e1-437e-9816-2a624d236ec2">
<img width="400" alt="image" src="https://github.com/josephjquinn/nnScratch/assets/81782398/5c7f234b-fe49-4ac0-b8d5-b764a869e502">

### Features

- Activation Functions: Supports ReLU, Sigmoid, and Leaky ReLU activation functions.
- Loss Function: Cross-entropy loss function for classification tasks.
- Initialization Methods: Supports initialization methods like random, He initialization, and normalized initialization.
- Training: Includes options for mini-batch training, with customizable batch size and learning rate.
- Monitoring: Real-time plots of training loss, validation loss, and accuracy during training.
- Prediction: Single and grid predictions on unseen data.
- Hyperparameter Optimization Script: Optimize and plot hyperparameters such as hidden layer neurons, learning rate, activation function, number of epochs, initialization method, and batch size.
- Saving and loading pretrained model weights.

## Design Notes


### Data & Network Structure 
![image](https://github.com/josephjquinn/nnScratch/assets/81782398/57fde786-7a1a-4798-b78a-e4431da9a1e2)

![image](https://github.com/josephjquinn/nnScratch/assets/81782398/84bc1149-2239-40df-9b9f-646b63046d78)


### Propagation Equations
![image](https://github.com/josephjquinn/nnScratch/assets/81782398/f05ff301-9c63-42cb-88dc-2da9710ca779)

![image](https://github.com/josephjquinn/nnScratch/assets/81782398/72abd4df-3288-4a1c-be9a-37928f9a6e74)

![image](https://github.com/josephjquinn/nnScratch/assets/81782398/8d5046a1-ddb8-4f39-a18a-5e378836d6b0)

### Optimization
![image](https://github.com/josephjquinn/nnScratch/assets/81782398/7622947d-75f7-4d8e-a74a-a9bf00eab3bb)

### Hyperparameters
![image](https://github.com/josephjquinn/nnScratch/assets/81782398/297e3e7b-86c7-425e-bb49-edfb02bb0baa)
![image](https://github.com/josephjquinn/nnScratch/assets/81782398/b9c59247-42c5-45b1-be85-2355879be7df)


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
- `main.py` Training script 
- `hyper.py` Calculates and saves optimal hyperparameters
