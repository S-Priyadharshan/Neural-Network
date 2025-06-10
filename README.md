# Custom Neural Network Implementations (C++ & CUDA)

This repository contains custom implementations of a neural network, showcasing different approaches to achieve varying levels of performance and functionality. The project is divided into three distinct sub-projects:

* **NeuralNetCPU**: A foundational implementation of a neural network in pure C++.
* **NeuralNetGPU**: An optimized version leveraging CUDA for GPU acceleration.
* **NeuralNetSFML**: An enhanced version that combines CUDA optimization with SFML for real-time visual representation of the neural network's behavior.

---

## Project Structure

The repository is organized into three main project directories, each focusing on a specific implementation:

* `NeuralNetCPU/`: Contains the C++ CPU-based neural network.
* `NeuralNetGPU/`: Houses the CUDA-optimized neural network.
* `NeuralNetSFML/`: Will contain the CUDA-optimized neural network with SFML visualization.

---

## NeuralNetCPU

This project provides a basic, object-oriented implementation of a feedforward neural network using standard C++. It's designed for clarity and understanding the fundamental concepts of neural network architecture, forward propagation, and backpropagation.

### Features

* **Feedforward Network**: Standard multi-layer perceptron.
* **Backpropagation**: Implementation of the backpropagation algorithm for training.
* **Sigmoid Activation**: Uses the sigmoid function for neuron activation.
* **Mean Squared Error**: Calculates error using the root mean square.
* **Training Data Handling**: Reads network topology and training data from a text file.

### Building and Running

1.  Navigate to the `NeuralNetCPU/` directory.
2.  Compile the source code using a C++ compiler (e.g., g++):
    ```bash
    g++ -o NeuralNetCPU main.cpp -std=c++11
    ```
3.  Run the executable:
    ```bash
    ./NeuralNetCPU
    ```
    Make sure `trainingData.txt` is present in the same directory as the executable.

### Example `trainingData.txt` format:

```
topology: 2 2 1
in: 0.0 0.0
out: 0.0
in: 0.0 1.0
out: 1.0
in: 1.0 0.0
out: 1.0
in: 1.0 1.0
out: 0.0
```

---

## NeuralNetGPU

This project significantly enhances the neural network's performance by offloading computationally intensive tasks to the GPU using NVIDIA's CUDA platform. This version focuses on demonstrating the speedup achievable through parallel processing.

### Features

* **CUDA Acceleration**: Utilizes GPU for parallel computation of neuron activations and matrix multiplications.
* **Custom Kernels**: Implements custom CUDA kernels for efficient feedforward operations.
* **Device Memory Management**: Handles memory allocation and transfer between host (CPU) and device (GPU).

### Building and Running

1.  Ensure you have the NVIDIA CUDA Toolkit installed and configured.
2.  Navigate to the `NeuralNetGPU/` directory.
3.  Compile the CUDA code using `nvcc` (NVIDIA CUDA Compiler):
    ```bash
    nvcc -o NeuralNetGPU main.cu TrainingData.cpp -std=c++11 -lcudart
    ```
    (Note: You might need to adjust the compilation command based on your specific CUDA setup and if `TrainingData.h` or `cuda_utils.cuh` require separate compilation or linking.)
4.  Run the executable:
    ```bash
    ./NeuralNetGPU
    ```
    Ensure `trainingData.txt` is available for the network to initialize.

---

## NeuralNetSFML

This project extends the `NeuralNetGPU` implementation by integrating the Simple and Fast Multimedia Library (SFML) to provide a visual representation of the neural network. This allows for a dynamic and intuitive understanding of how the network processes data and learns.

### Features

* **Real-time Visualization**: Displays the network's structure, activation levels, and potentially training progress.
* **SFML Graphics**: Leverages SFML for rendering the visual interface.
* **CUDA Integration**: Maintains the performance benefits of GPU acceleration.

### Building and Running

(Details for building and running the `NeuralNetSFML` project will be added once its development is complete.)

---

## Contribution

Feel free to explore the code, open issues, or submit pull requests. Any contributions to improve the implementations, add new features, or enhance documentation are welcome!


