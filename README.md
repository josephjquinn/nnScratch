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

## Design Notes


### Data & Network Structure 
![image](https://github.com/josephjquinn/nnScratch/assets/81782398/57fde786-7a1a-4798-b78a-e4431da9a1e2)
<img width="700" alt="data" src="https://github.com/josephjquinn/nnScratch/assets/81782398/491bdc25-d86c-46cb-8ab6-2933de84b4f6">

![image](https://github.com/josephjquinn/nnScratch/assets/81782398/84bc1149-2239-40df-9b9f-646b63046d78)

<img width="700" alt="netstructure" src="https://github.com/josephjquinn/nnScratch/assets/81782398/100f17ea-01c9-46eb-b0f4-ead7c809e311">

### Propagation Equations
![image](https://github.com/josephjquinn/nnScratch/assets/81782398/f05ff301-9c63-42cb-88dc-2da9710ca779)
<img width="700" alt="backprop" src="https://github.com/josephjquinn/nnScratch/assets/81782398/2f602d65-bfbe-4932-b72d-1236edff12f8">

![image](https://github.com/josephjquinn/nnScratch/assets/81782398/72abd4df-3288-4a1c-be9a-37928f9a6e74)
<img width="700" alt="calc" src="https://github.com/josephjquinn/nnScratch/assets/81782398/454b8ad9-5bf6-464f-aa76-d2023e987208">

![image](https://github.com/josephjquinn/nnScratch/assets/81782398/8d5046a1-ddb8-4f39-a18a-5e378836d6b0)
<img width="700" alt="gradient" src="https://github.com/josephjquinn/nnScratch/assets/81782398/3f02f8bb-e08f-447c-8ee1-f525df7162c0">

### Optimization
![image](https://github.com/josephjquinn/nnScratch/assets/81782398/7622947d-75f7-4d8e-a74a-a9bf00eab3bb)
<img width="700" alt="optimi" src="https://github.com/josephjquinn/nnScratch/assets/81782398/241186bc-f740-440d-ae8b-595646484c4b">

### Hyperparameters
![image](https://github.com/josephjquinn/nnScratch/assets/81782398/297e3e7b-86c7-425e-bb49-edfb02bb0baa)
<img width="700" alt="hyperparams" src="https://github.com/josephjquinn/nnScratch/assets/81782398/69164a68-8e8b-46c4-b5a3-519731be5293">
![image](https://github.com/josephjquinn/nnScratch/assets/81782398/b9c59247-42c5-45b1-be85-2355879be7df)
<img width="700" alt="hypergraph" src="https://github.com/josephjquinn/nnScratch/assets/81782398/2c5a6016-d258-4846-b96c-d3a410c1aec0">


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
