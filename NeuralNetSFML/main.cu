#include<iostream>
#include"SFML_includes.cuh"
#include"TrainingData.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "device_atomic_functions.h"
#include <random>
using namespace std;

__device__ float activationFunction(float x) {
	return 1.0f / (1.0f + expf(-x));
}

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

struct Layer {
	int numInputs;
	int numNeurons;
	vector<float>weights;
	vector<float>bias;
	vector<float>output;

	float* d_weights = nullptr;
	float* d_bias = nullptr;
	float* d_output = nullptr;

	void Initialize(int inputSize, int neuronCount);
	void feedForward(float* d_input);
	void allocateOnDevice();
	void freeDeviceMem();
};

float randWeight(void) { return static_cast<float>(rand()) / RAND_MAX; }


void Layer::Initialize(int inputSize, int neuronCount) {
	numInputs = inputSize;
	numNeurons = neuronCount;

	weights.resize(numInputs * numNeurons);
	bias.resize(numNeurons);
	output.resize(numNeurons);

	for (int n = 0;n < numNeurons;n++) {
		for (int i = 0;i < numInputs;i++) {
			weights[n * numInputs + i] = randWeight();
		}
		bias[n] = randWeight();
		output[n] = 0.0f;
	}
}

void Layer::allocateOnDevice() {
	HANDLE_ERROR(cudaMalloc(&d_weights, sizeof(float) * weights.size()));
	HANDLE_ERROR(cudaMalloc(&d_bias, sizeof(float) * bias.size()));
	HANDLE_ERROR(cudaMalloc(&d_output, sizeof(float) * output.size()));

	HANDLE_ERROR(cudaMemcpy(d_weights, weights.data(), sizeof(float) * weights.size(), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_bias, bias.data(), sizeof(float) * bias.size(), cudaMemcpyHostToDevice));
}

void Layer::freeDeviceMem() {
	HANDLE_ERROR(cudaFree(d_weights));
	HANDLE_ERROR(cudaFree(d_bias));
	HANDLE_ERROR(cudaFree(d_output));
}

void Layer::feedForward(float* d_input) {
	int blockSize = numInputs;
	feedForwardKernel << <numNeurons, blockSize, blockSize * sizeof(float) >> > (d_input, d_weights, d_bias, d_output, numInputs);
}

struct Net {
	vector<Layer> layers;

	void Initialize(vector<unsigned>& topology);
	vector<float> feedForward(vector<float>& inputVals);
	vector<vector<float>> getAllWeights() const;
	vector<vector<float>> getAllBiases() const;
	vector<vector<float>> getAllActivations() const;

};

void Net::Initialize(vector<unsigned>& topology) {
	layers.clear();

	for (int i = 0;i < topology.size() - 1;i++) {
		Layer layer;
		layer.Initialize(topology[i], topology[i + 1]);
		layers.push_back(layer);
	}
}

vector<vector<float>> Net::getAllWeights() const{
	vector<vector<float>>allWeights;
	
	for (const auto& layer : layers) {
		allWeights.push_back(layer.weights);
	}

	return allWeights;
}

vector<vector<float>> Net::getAllBiases() const {
	vector<vector<float>> allBiases;
	for (const auto& layer : layers) {
		allBiases.push_back(layer.bias);
	}
	return allBiases;
}

vector<vector<float>> Net::getAllActivations() const {
	vector<vector<float>> allActivations;
	for (const auto& layer : layers) {
		allActivations.push_back(layer.output);
	}
	return allActivations;
}

int main() {
	TrainingData data("trainingData.txt");

	vector<unsigned>topology;
	data.getTopology(topology);

	Net net;
	net.Initialize(topology);

	for (Layer& layer : net.layers) {
		layer.allocateOnDevice();
	}

	cout << "Network initialized with " << topology.size() << " layers." << endl;

	vector<float>inputVals, outputVals;

	if (!data.getNextInputs(inputVals)) {
		cerr << "Failed to read training data" << endl;
		return 1;
	}

	float* d_input = nullptr;
	HANDLE_ERROR(cudaMalloc(&d_input, sizeof(float) * inputVals.size()));

	sf::RenderWindow window(sf::VideoMode(800, 600), "Neural Network");
	window.setFramerateLimit(60);

	sf::Font font;
	if (!font.loadFromFile("Monospace.ttf")) {
		std::cerr << "Font load failed\n";
		return -1;
	}

	int SCREEN_WIDTH = 800;
	int SCREEN_HEIGHT = 600;

	vector<vector<sf::Vector2f>> positions(topology.size());
	for (unsigned int i = 0; i < topology.size(); i++) {
		float x = (SCREEN_WIDTH/ (topology.size() + 1)) * (i + 1);
		for (unsigned int j = 0; j < topology[i]; j++) {
			float y = (SCREEN_HEIGHT / (topology[i] + 1)) * (j + 1);
			positions[i].emplace_back(x, y);
		}
	}

	for (int epoch = 0;epoch < 10;epoch++) {
		cout << "Epoch: " << epoch << "\n";

		HANDLE_ERROR(cudaMemcpy(d_input, inputVals.data(), sizeof(float) * inputVals.size(), cudaMemcpyHostToDevice));

		float* currInput = d_input;

		for (int l = 0;l < net.layers.size();l++) {
			Layer& layer = net.layers[l];
			layer.feedForward(currInput);

			HANDLE_ERROR(cudaMemcpy(layer.output.data(), layer.d_output, sizeof(float) * layer.output.size(), cudaMemcpyDeviceToHost));

			currInput = layer.d_output;

		}

		auto activations = net.getAllActivations();

		sf::Event e;
		while (window.pollEvent(e)) {
			if (e.type == sf::Event::Closed)window.close();
			if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))window.close();
		}

		/*vector<vector<float>> allWeights = net.getAllWeights();
		vector<vector<float>> allBiases = net.getAllBiases();
		vector<vector<float>> allActivations = net.getAllActivations();*/
		//drawNeuralNet(topology, allWeights, allBiases, allActivations);
		drawNeuralNet(window, font, positions, topology, activations);
		sf::sleep(sf::milliseconds(500));
	}

	for (Layer& layer : net.layers) {
		layer.freeDeviceMem();
	}
	cudaFree(d_input);
	return 0;

}