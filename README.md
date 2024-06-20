# Fully Connected Neural Network using Numpy & Calculus

This project was a learning project for me to better understand the fundamentals of Neural Networks and their architecture. I created this class without using any machine learning libraries doing most of the calculations by hand.
This repository contains a simple neural network implementation in Python, designed for educational purposes. The neural network supports customizable architecture and training options.

<img width="400" src="https://github.com/josephjquinn/nnScratch/assets/81782398/6a94a5a4-8d7e-47f3-846b-50144b225212">
<img width="400" src="https://github.com/josephjquinn/nnScratch/assets/81782398/6cef7511-62c5-4e66-a095-4fb96ece4a39">

<img width="1400" alt="image" src="https://github.com/josephjquinn/nnScratch/assets/81782398/6d4ea186-1baf-44cf-96c3-cfda0a8138dc">

![f123](https://github.com/josephjquinn/nnScratch/assets/81782398/b53781f5-bf3f-4eab-8af7-260971705ac6)
![f21](https://github.com/josephjquinn/nnScratch/assets/81782398/691a4646-fa72-4669-9dc3-293ff20ebd7a)

### Results
<img width="750" src="https://github.com/josephjquinn/nnScratch/assets/81782398/21b95943-6276-4d1f-81c8-d9eb669ccdc9">
<img width="750" src="https://github.com/josephjquinn/nnScratch/assets/81782398/5f8d7c30-fbdf-4c69-986f-0667a4738ff4">
<img width="750" src="https://github.com/josephjquinn/nnScratch/assets/81782398/68f93074-4f99-4bdb-9549-b5ce88931c5a">
<img width="750" src="https://github.com/josephjquinn/nnScratch/assets/81782398/bdf1ab15-ebc7-4cc7-9411-9cd47af2b7ad">
<img width="750" src="https://github.com/josephjquinn/nnScratch/assets/81782398/74da8414-bbf4-4dc6-a85a-19a083c8f202">
<img width="750" src="https://github.com/josephjquinn/nnScratch/assets/81782398/bd986644-c537-4c7c-8948-ff186da89b1c">
<img width="750" src="https://github.com/josephjquinn/nnScratch/assets/81782398/1c486eb5-f128-4749-9d9f-a319372621e4">
<img width="750" src="https://github.com/josephjquinn/nnScratch/assets/81782398/c011c8e2-920d-42e5-a3b5-ab6bf8a55d94">
<img width="750" src="https://github.com/josephjquinn/nnScratch/assets/81782398/09d7559d-c733-4d5a-92a5-e745463e9aeb">

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
