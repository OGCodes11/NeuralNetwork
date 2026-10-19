# Custom Multi-Layer Neural Network for Text Classification

A custom neural network built completely from scratch in Java without external machine learning libraries. This project was developed as part of a coursework lab to classify text files based on writing style (specifically distinguishing between texts written by William Shakespeare and Taylor Swift).

PyTorch version: see the pytorch folder, a rebuild of this network with PyTorch

## Project Architecture

The project consists of the following core components:

* **`RELUNeuron.java`**: Implements individual hidden layer neurons using the Rectified Linear Unit (ReLU) activation function and handles weight initialization, forward passing, and binary serialization.
* **`LogisticNeuron.java`**: Implements the final output layer neuron using the Logistic (Sigmoid) activation function and gradient descent training updates.
* **`NeuralLayer.java`**: Manages a collection of neurons operating together as a unified network layer.
* **`NeuralNetwork.java`**: Ties the hidden layers and output neuron together, handling forward propagation, training steps, and binary file input/output (`read`/`write`).
* **`TrainNN.java`**: Executable program that reads a directory of training files, converts text into input vectors, trains the network, and saves the model weights.
* **`UseNN.java`**: Executable program that loads a trained model file, processes an unseen test file, and predicts the author with a percentage confidence score.

### Compilation
Open your terminal in the project source directory and compile all Java files:
```bash
javac *.java

