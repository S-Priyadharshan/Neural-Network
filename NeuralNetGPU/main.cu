#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "device_atomic_functions.h"
#include "TrainingData.h"
#include <random>
#include <cassert>

#include"cuda_utils.cuh";

using namespace std;

__device__ float activationFunction(float x) {
	return 1.0f / (1.0f + expf(-x));
}

__global__ void feedForwardKernel(
	const float* inputs,
	const float* weights,
	const float* bias,
	float *outputs,
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

	for (int s = blockDim.x / 2;s > 0;s>>= 1) {
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

	weights.resize(numNeurons * numInputs);
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
};

void Net::Initialize(vector<unsigned>& topology) {
	layers.clear();

	for (int i = 0;i < topology.size()-1;i++) {
		Layer layer;
		layer.Initialize(topology[i], topology[i + 1]);
		layers.push_back(layer);
	}
}

int main() {
	srand(static_cast<unsigned>(time(0)));

	TrainingData data("trainingData.txt");

	vector<unsigned>topology;
	data.getTopology(topology);

	Net net;
	net.Initialize(topology);


	//for (int l = 0; l < net.layers.size(); ++l) {
	//	cout << "Layer " << l << ":\n";

	//	const Layer& layer = net.layers[l];

	//	// Print weights
	//	cout << "  Weights:\n";
	//	for (int n = 0; n < layer.numNeurons; ++n) {
	//		cout << "    Neuron " << n << ": ";
	//		for (int i = 0; i < layer.numInputs; ++i) {
	//			cout << layer.weights[n * layer.numInputs + i] << " ";
	//		}
	//		cout << endl;
	//	}

	//	// Print biases
	//	cout << "  Biases:\n    ";
	//	for (int n = 0; n < layer.numNeurons; ++n) {
	//		cout << layer.bias[n] << " ";
	//	}
	//	cout << "\n\n";
	//}

	for (Layer& layer : net.layers) {
		layer.allocateOnDevice();
	}

	cout << "Network initialized with " << topology.size() << " layers." << endl;
		
	vector<float>inputVals, outputVals;

	/*while (!data.isEof()) {
		if (!data.getNextInputs(inputVals))break;
		if (!data.getTargetOutputs(outputVals))break;

		assert(inputVals.size() == topology[0]);
		assert(outputVals.size() == topology.back());
	}*/

	if (!data.getNextInputs(inputVals)) {
		cerr << "Failed to read training data" << endl;
		return 1;
	}

	float* d_input = nullptr;
	HANDLE_ERROR(cudaMalloc(&d_input, sizeof(float) * inputVals.size()));

	for (int epoch = 0;epoch < 10;epoch++) {
		cout << "Epoch: " << epoch << "\n";

		HANDLE_ERROR(cudaMemcpy(d_input, inputVals.data(), sizeof(float) * inputVals.size(), cudaMemcpyHostToDevice));

		float* currInput = d_input;

		for (int l = 0;l< net.layers.size();l++) {
			Layer& layer = net.layers[l];
			layer.feedForward(currInput);

			HANDLE_ERROR(cudaMemcpy(layer.output.data(), layer.d_output, sizeof(float) * layer.output.size(), cudaMemcpyDeviceToHost));

			currInput = layer.d_output;

			cout << "  Layer " << l << " Outputs: ";
			for (float val : layer.output) {
				cout << val << " ";
			}
			cout << "\n";
		}
		/*for (int l = 0; l < net.layers.size(); ++l) {
			cout << "Layer " << l << " Weights:\n";
			Layer& layer = net.layers[l];
			for (int n = 0; n < layer.numNeurons; ++n) {
				cout << "  Neuron " << n << ": ";
				for (int i = 0; i < layer.numInputs; ++i) {
					cout << layer.weights[n * layer.numInputs + i] << " ";
				}
				cout << "\n";
			}
		}*/
	}

	for (Layer& layer : net.layers) {
		layer.freeDeviceMem();
	}
	cudaFree(d_input);
	return 0;
}