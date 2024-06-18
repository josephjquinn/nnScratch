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
<img width="700" alt="data" src="https://github.com/josephjquinn/nnScratch/assets/81782398/491bdc25-d86c-46cb-8ab6-2933de84b4f6">
<img width="700" alt="netstructure" src="https://github.com/josephjquinn/nnScratch/assets/81782398/100f17ea-01c9-46eb-b0f4-ead7c809e311">

### Propagation Equations
<img width="700" alt="backprop" src="https://github.com/josephjquinn/nnScratch/assets/81782398/2f602d65-bfbe-4932-b72d-1236edff12f8">
<img width="700" alt="calc" src="https://github.com/josephjquinn/nnScratch/assets/81782398/454b8ad9-5bf6-464f-aa76-d2023e987208">
<img width="700" alt="gradient" src="https://github.com/josephjquinn/nnScratch/assets/81782398/3f02f8bb-e08f-447c-8ee1-f525df7162c0">

### Optimization

<img width="700" alt="optimi" src="https://github.com/josephjquinn/nnScratch/assets/81782398/241186bc-f740-440d-ae8b-595646484c4b">

### Hyperparameters
<img width="700" alt="hyperparams" src="https://github.com/josephjquinn/nnScratch/assets/81782398/69164a68-8e8b-46c4-b5a3-519731be5293">
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
