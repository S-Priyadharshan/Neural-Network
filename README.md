# Custom Implementation of Artificial Neural Network ( C++ | CUDA | SFML )

Hey There!

## Abstract

This repository contains my project which implements a fully functional Feedforward Neural Network with both CPU and CUDA-accelerated GPU backend, coupled with an SFML based Real-time visualizer.
It provides an interactive and intuitive approach to understanding how Neural Networks work on a fundamental level and implements many of its core concepts from scratch.

The code is modularized into the following:

* **NeuralNetCPU**: A foundational implementation of a neural network in pure C++.

* **NeuralNetGPU**: An optimized version leveraging CUDA for GPU acceleration.
  
* **NeuralNetSFML**: An enhanced version that combines CUDA optimization with SFML for real-time visual representation of the neural network's behavior.

---

## Index

1. CPU Neural Network

2. CUDA Neural Network

3. SFML Visualizer

## 1. NeuralNetCPU

The CPU implementation builds a fully-connected feedforward neural net using classes like Net, Layer and Neuron.
It implements all the features from scratch and handles all the required low level mathematical operations.

### Features

* **Feedforward Network**: Standard multi-layer perceptron.
* **Backpropagation**: Implementation of the backpropagation algorithm for training.
* **Sigmoid Activation**: Uses the sigmoid function for neuron activation.
* **Mean Squared Error**: Calculates error using the root mean square.
* **Training Data Handling**: Reads network topology and training data from a text file.
* **Object Oriented Design**: The network is modularized into components appropriate to OOPs design principles

### Examples

This is a successful example of teaching the model how an XOR operations works and allowing it to identify the non-linear relationship between the input and output variables

<img width="618" height="515" alt="image" src="https://github.com/user-attachments/assets/3ca08103-484d-453c-806f-5dc8312c3f66" />

Structure Code Snippet:

```c++
class Neuron {
public:
	Neuron(unsigned numOutputs, unsigned Index); // here num outputs is like the amount of neurons in the next layer
	void setOutputVal(double val) { outputVal = val; }
	double getOutputVal() const { return outputVal; }
	void feedForward(const Layer& prevLayer);
	void calcOutputGradients(double targetVal);
	void calcHiddenGradient(const Layer& nextLayer);
	void updateInputWeights(Layer& prevLayer);

private:
	double outputVal;
	unsigned n_Index;
	static double randWeight(void) { return static_cast<double>(rand()) / RAND_MAX; }
	vector<Connection> outputWeights;
	double sumDow(const Layer& nextLayer);
	static double transferFunction(double x);
	static double transferFunctionDerivative(double x);
	double gradient;
	static double eta;
	static double alpha;
};

class Net {
public:
	Net(const vector<unsigned>& topology);//constructor
	void feedForward(const vector<double>& inputVals);
	void backProp(const vector<double>& targetVals);
	void getResults(vector<double>& resultVals)const;
	double getRecentAverageError(void)const { return avgError; }

private:
	vector<Layer> layers; // this is vector<vector<Neuron>> layers
	double error;         // basically layout of the whole network
	double avgError;
	static double avgErrorSmoothingFactor;
};
```
---

## NeuralNetGPU

This project significantly enhances the neural network's performance by offloading computationally intensive tasks to the GPU using NVIDIA's CUDA platform.
The GPU version accelerates forward and backward propagation using CUDA kernels and all major operations like dot product and gradient updates are done on the GPU.

### Features

* **CUDA Acceleration**: Utilizes GPU for parallel computation of neuron activations and matrix multiplications.
* **Custom Kernels**: Implements custom CUDA kernels for efficient feedforward operations.
* **Device Memory Management**: Handles memory allocation and transfer between host (CPU) and device (GPU).

### Example

The same test run on a CUDA accelerated Neural network.

<img width="299" height="436" alt="image" src="https://github.com/user-attachments/assets/3db39394-77a0-43eb-8f87-2d0229b303e7" />

The performance hit is due to the data transfer between the CPU and GPU and it is optimized to work better with larger datasets and more intensive machine learning processes.

Code Snippet:

```c++
__global__ void feedForwardKernel(
	const float* inputs,
	const float* weights,
	const float* bias,
	float* outputs,
	int numInputs
)
{
	extern __shared__ float cache[];

	int Nid = blockIdx.x;
	int tid = threadIdx.x;

	float product = 0.0f;

	if (tid < numInputs) {
		product = inputs[tid] * weights[Nid * numInputs + tid];
	}

	cache[tid] = product;
	__syncthreads();

	for (int s = blockDim.x / 2;s > 0;s >>= 1) {
		if (tid < s && (tid + s) < numInputs) {
			cache[tid] += cache[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0) {
		float sum = cache[0] + bias[Nid];
		outputs[Nid] = activationFunction(sum);
	}
}
```

---

## NeuralNetSFML

Perhaps the most interesting part of my nerual network implementation was learning how to implement SFML graphic libraries into helping visualize Neural Networks and understand how they mirror the human brain and actually learn stuff.

### Features

* **Real-time Visualization**: Displays the network's structure, activation levels, and potentially training progress.
* **SFML Graphics**: Leverages SFML for rendering the visual interface.
* **CUDA Integration**: Maintains the performance benefits of GPU acceleration.
* **Interactive**: Place RGB points along the screen and watch the model learn to differentiate them.

### Example

This is how the neural network is presented in its training stage

<img width="792" height="595" alt="image" src="https://github.com/user-attachments/assets/5e99e345-75d0-443e-abae-e687247dccdd" />

and this is an example of it learning 

<img width="1598" height="639" alt="image" src="https://github.com/user-attachments/assets/5bf0efde-6907-40d1-bfc2-4c277b39c111" />

<img width="1594" height="635" alt="image" src="https://github.com/user-attachments/assets/71f757e7-17f5-4f33-9eea-477e72d8fd49" />

<img width="1600" height="631" alt="image" src="https://github.com/user-attachments/assets/6fd21eff-2763-4257-aa81-fb0c7b263bad" />

Code Snippet:

```c++
void drawNeuralNet(sf::RenderWindow& window, sf::Font& font, vector<vector<sf::Vector2f>>& positions, vector<unsigned>& topology, vector<vector<float>>& activations,vector<float>&inputVals) {
	
	int CIRCLE_RADIUS = 32.f;

	window.clear(sf::Color::Black);

    for (size_t i = 0; i < positions.size() - 1; i++) {
        for (const auto& from : positions[i]) {
            for (const auto& to : positions[i + 1]) {
                sf::Vertex line[] = {
                    sf::Vertex(from, sf::Color::White),
                    sf::Vertex(to, sf::Color::White)
                };
                window.draw(line, 2, sf::Lines);
            }
        }
    }

    for (size_t i = 0; i < positions.size(); i++) {
        for (size_t j = 0; j < positions[i].size(); j++) {
            const auto& pos = positions[i][j];

            sf::CircleShape neuron(32.f);
            neuron.setOrigin(32.f, 32.f);
            neuron.setPosition(pos);
            neuron.setFillColor(sf::Color::White);
            window.draw(neuron);

            sf::Text text;
            text.setFont(font);
            text.setCharacterSize(14);
            text.setFillColor(sf::Color::Black);
            std::ostringstream ss;
            if (i == 0 && j<inputVals.size())
                ss << std::fixed << std::setprecision(2) <<inputVals[j];
            else
                ss << std::fixed << std::setprecision(2) << activations[i - 1][j];

            text.setString(ss.str());
            sf::FloatRect bounds = text.getLocalBounds();
            text.setOrigin(bounds.width / 2, bounds.height / 2 + bounds.top);
            text.setPosition(pos);
            window.draw(text);
        }
    }
}

```

---

## References

These resources were absolutely crucial in helping me making this project so please do check them out!

Neural Net:

https://www.youtube.com/watch?v=sK9AbJ4P8ao
https://millermattson.com/dave/?p=54

SFML Integration:

https://www.youtube.com/watch?v=Zrrnqd0rCXg
https://github.com/Kofybrek/Neural-network/tree/Main

Thank you so much for going through the repo and hit me up if you want to collaborate on more interesting projects like these.
