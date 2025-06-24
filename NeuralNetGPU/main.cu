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

__device__ float activationFunctionDerivative(float x) {
	return x * (1.0 - x);
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

__global__ void computeDelta(
	const float* predicted,
	const float* target,
	float* errors,
	int numOutputNeurons
) {
	int tid = threadIdx.x;
	float error = 0.0;
	if (tid < numOutputNeurons) {
		errors[tid] = target[tid] - predicted[tid];
	}
}

__global__ void computeOutputGradients(
	const float* outputVals,
	const float* errors,
	float* gradients,
	int numNeurons
) {
	int tid = threadIdx.x;

	if (tid < numNeurons) {
		gradients[tid] = errors[tid] * activationFunctionDerivative(outputVals[tid]);	
	}

}

__global__ void computeHiddenLayerGradients(
	const float* hiddenOutputs,
	const float* nextWeights,
	const float* nextGradients,
	float* hiddenGradients,
	int numHiddenNeurons,
	int numNextNeurons
) {
	int tid = threadIdx.x;

	if (tid < numHiddenNeurons) {
		float sum = 0.0f;

		for (int j = 0;j < numNextNeurons;j++) {
			sum += nextWeights[j * numHiddenNeurons + tid] * nextGradients[j];
		}

		hiddenGradients[tid] = sum * activationFunctionDerivative(hiddenOutputs[tid]);
	}
}

__global__ void updateWeights(
	float* weights,
	const float* inputs,
	const float* gradients,
	float* prevDeltaWeights,
	int numInputs,
	int numNeurons,
	float eta,
	float alpha
) {
	int neuronIdx = blockIdx.x;
	int weightIdx = threadIdx.x;

	if (neuronIdx < numNeurons && weightIdx < numInputs) {
		int index = neuronIdx * numInputs + weightIdx;
		float oldDelta = prevDeltaWeights[index];
		float newDelta = eta * inputs[weightIdx] * gradients[neuronIdx] + alpha * oldDelta;

		prevDeltaWeights[index] = newDelta;
		weights[index] += newDelta;
	}
}

__global__ void computeRMSError(
	const float* deltas,
	float* rms,
	int size
) {
	extern __shared__ float cache[];
	int tid = threadIdx.x;
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	float temp = 0.0f;
	if (i < size) {
		temp = deltas[i] * deltas[i];
	}
	
	cache[tid] = temp;
	__syncthreads();

	for (int s = blockDim.x / 2;s > 0;s >>= 1) {
		if (tid < s) {
			cache[tid] += cache[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0) {
		atomicAdd(rms, cache[0]);
	}
}

__global__ void updateBias(float* bias, const float* gradients, int numNeurons, float eta) {
	int tid = threadIdx.x;
	if (tid < numNeurons) {
		bias[tid] += eta * gradients[tid];
	}
}


struct Layer {
	int numNeurons;
	int numInputs;
	vector<float>weights;
	vector<float>bias;
	vector<float>output;
	vector<float>delta;
	vector<float>gradients;
	vector<float>prevDeltaWeights;

	float* d_weights = nullptr;
	float* d_bias = nullptr;
	float* d_output = nullptr;
	float* d_delta = nullptr;
	float* d_gradients = nullptr;
	float* d_prevDeltaWeights = nullptr;

	void Initialize(int inputSize, int neuronCount);
	void feedForward(float* d_input);
	void backProp(float* d_targetVals);
	void allocateOnDevice();
	void freeDeviceMem();
};

//float randWeight(void) { return static_cast<float>(rand()) / RAND_MAX; }
float randWeight() { return 2.0f * static_cast<float>(rand()) / RAND_MAX - 1.0f; }


void Layer::Initialize(int inputSize, int neuronCount) {
	numInputs = inputSize;
	numNeurons = neuronCount;

	weights.resize(numNeurons * numInputs);
	bias.resize(numNeurons);
	output.resize(numNeurons);
	delta.resize(numNeurons, 0.0f);
	gradients.resize(numNeurons, 0.0f);
	prevDeltaWeights.resize(numNeurons * numInputs, 0.0f);

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
	HANDLE_ERROR(cudaMalloc(&d_delta, sizeof(float) * numNeurons));
	HANDLE_ERROR(cudaMalloc(&d_gradients, sizeof(float) * numNeurons));
	HANDLE_ERROR(cudaMalloc(&d_prevDeltaWeights, sizeof(float) * prevDeltaWeights.size()));

	HANDLE_ERROR(cudaMemcpy(d_weights, weights.data(), sizeof(float) * weights.size(), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_bias, bias.data(), sizeof(float) * bias.size(), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_delta, delta.data(), sizeof(float) * numNeurons, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_gradients, gradients.data(), sizeof(float) * numNeurons, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_prevDeltaWeights, prevDeltaWeights.data(), sizeof(float) * prevDeltaWeights.size(), cudaMemcpyHostToDevice));

}

void Layer::freeDeviceMem() {
	HANDLE_ERROR(cudaFree(d_weights));
	HANDLE_ERROR(cudaFree(d_bias));
	HANDLE_ERROR(cudaFree(d_output));
	HANDLE_ERROR(cudaFree(d_delta));
	HANDLE_ERROR(cudaFree(d_gradients));
	HANDLE_ERROR(cudaFree(d_prevDeltaWeights));
}

void Layer::feedForward(float* d_input) {
	int blockSize = numInputs;
	feedForwardKernel << <numNeurons, blockSize, blockSize * sizeof(float) >> > (d_input, d_weights, d_bias, d_output, numInputs);
}	


struct Net {
	vector<Layer> layers;
	
	void Initialize(vector<unsigned>& topology);
	vector<float> feedForward(vector<float>& inputVals);
	vector<float> backProp(vector<float>& targetVals);
};

void Net::Initialize(vector<unsigned>& topology) {
	layers.clear();

	for (int i = 0;i < topology.size()-1;i++) {
		Layer layer;
		layer.Initialize(topology[i], topology[i + 1]);
		layers.push_back(layer);
	}
}

void showVectorVals(string label, vector<float>& v)
{
	cout << label << " ";
	for (unsigned i = 0; i < v.size(); ++i) {
		cout << v[i] << " ";
	}

	cout << endl;
}


int main() {
	srand(static_cast<unsigned>(time(0)));

	TrainingData data("trainingData.txt");

	vector<unsigned>topology;
	data.getTopology(topology);

	Net net;
	net.Initialize(topology);

	for (Layer& layer : net.layers) {
		layer.allocateOnDevice();
	}

	cout << "Network initialized with " << topology.size() << " layers." << endl;
		
	vector<float>inputVals, outputVals, targetVals;

	if (!data.getNextInputs(inputVals)) {
		cerr << "Failed to read training data" << endl;
		return 1;
	}

	if (!data.getTargetOutputs(outputVals)) {
		cerr << "Failed to read training data" << endl;
		return 1;
	}

	float eta = 0.1f;
	float alpha = 0.5f;
	float* d_input = nullptr;
	float* d_targetVals = nullptr;
	float* d_rms = nullptr;
	HANDLE_ERROR(cudaMalloc(&d_input, sizeof(float) * inputVals.size()));
	HANDLE_ERROR(cudaMalloc(&d_targetVals, sizeof(float) * outputVals.size()));
	HANDLE_ERROR(cudaMalloc(&d_rms,sizeof(float)));

	HANDLE_ERROR(cudaMemcpy(d_input, inputVals.data(), sizeof(float)*inputVals.size(), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_targetVals, outputVals.data(), sizeof(float) * outputVals.size(), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemset(d_rms, 0, sizeof(float)));

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	for (int epoch = 0;epoch < 5000;epoch++) {
		data.reset();
		data.getTopology(topology);
		while (!data.isEof()) {
			//int N = 10;  // Print weights every N epochs
			//if ((epoch + 1) % N == 0) {
			//	std::cout << "\nEpoch " << epoch + 1 << " - Layer Weights:\n";
			//	for (int l = 0; l < net.layers.size(); ++l) {
			//		Layer& layer = net.layers[l];
			//		std::vector<float> h_weights(layer.weights.size());
			//		HANDLE_ERROR(cudaMemcpy(h_weights.data(), layer.d_weights, sizeof(float) * h_weights.size(), cudaMemcpyDeviceToHost));

			//		std::cout << "  Layer " << l << " Weights:\n";
			//		for (int n = 0; n < layer.numNeurons; ++n) {
			//			std::cout << "    Neuron " << n << ": ";
			//			for (int i = 0; i < layer.numInputs; ++i) {
			//				std::cout << h_weights[n * layer.numInputs + i] << " ";
			//			}
			//			std::cout << "\n";
			//		}
			//	}
			//}
			if (!data.getNextInputs(inputVals)) break;
			if (!data.getTargetOutputs(outputVals))break;
			if (inputVals.size() != topology[0]) break;
			
			cudaMemcpy(d_input, inputVals.data(), sizeof(float) * inputVals.size(), cudaMemcpyHostToDevice);
			cudaMemcpy(d_targetVals, outputVals.data(), sizeof(float) * outputVals.size(), cudaMemcpyHostToDevice);

			float* currInput = d_input;

			for (int l = 0;l < net.layers.size();l++) {
				Layer& layer = net.layers[l];
				layer.feedForward(currInput);
				currInput = layer.d_output;
			}

			Layer& outputLayer = net.layers.back();
			int numOutputNeurons = outputLayer.numNeurons;

			computeDelta << <1, numOutputNeurons >> > (
				outputLayer.d_output,
				d_targetVals,
				outputLayer.d_delta,
				numOutputNeurons
				);

			computeRMSError << <(numOutputNeurons + 255) / 256, 256, 256 * sizeof(float) >> > (
				outputLayer.d_delta,
				d_rms,
				numOutputNeurons
				);

			/*if ((epoch + 1) % 500 == 0) {
				float h_rms = 0.0f;
				HANDLE_ERROR(cudaMemcpy(&h_rms, d_rms, sizeof(float), cudaMemcpyDeviceToHost));
				h_rms = sqrtf(h_rms / numOutputNeurons);
				cout << "Epoch " << epoch << " RMS Error: " << h_rms << "\n";
				cudaMemset(d_rms, 0, sizeof(float));
			}*/


			computeOutputGradients << <1, numOutputNeurons >> > (outputLayer.d_output,
				outputLayer.d_delta,
				outputLayer.d_gradients,
				numOutputNeurons
				);

			
			for (int i = net.layers.size() - 2;i >= 0;i--) {
				Layer& hiddenLayer = net.layers[i];
				Layer& nextLayer = net.layers[i + 1];
				int numHidden = hiddenLayer.numNeurons;
				int numNext = nextLayer.numNeurons;
				computeHiddenLayerGradients << <1, numHidden >> > (
					hiddenLayer.d_output,
					nextLayer.d_weights,
					nextLayer.d_gradients,
					hiddenLayer.d_gradients,
					numHidden,
					numNext
					);
			}

			for (int i = net.layers.size() - 1; i >= 0; i--) {
				Layer& currLayer = net.layers[i];

				const float* prevOutputs;
				int prevSize;

				if (i == 0) {
					prevOutputs = d_input; 
					prevSize = 2;
				}
				else {
					prevOutputs = net.layers[i - 1].d_output;
					prevSize = net.layers[i - 1].numNeurons;
				}

				updateWeights << <currLayer.numNeurons, prevSize >> > (
					currLayer.d_weights,
					prevOutputs,
					currLayer.d_gradients,
					currLayer.d_prevDeltaWeights,
					prevSize,
					currLayer.numNeurons,
					eta,
					alpha
					);

				updateBias << <1, currLayer.numNeurons >> > (
					currLayer.d_bias,
					currLayer.d_gradients,
					currLayer.numNeurons,
					eta
					);
			}


		}
		
	}
	
	cudaEventRecord(stop);
	float time;
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	float* errors = (float*)malloc(sizeof(float) * outputVals.size());
	HANDLE_ERROR(cudaMemcpy(errors, net.layers.back().d_delta, sizeof(float) * outputVals.size(), cudaMemcpyDeviceToHost));

	cout << "Time taken for Training: " << time / 1000 << " s" << "\n";

	data.reset();
	data.getTopology(topology);

	cout << "\nFinal Outputs:\n";

	vector<float> resultVals(topology.back());

	d_input = nullptr;
	HANDLE_ERROR(cudaMalloc(&d_input, sizeof(float) * topology[0]));

	Layer& finalLayer = net.layers.back();

	while (!data.isEof()) {
		if (!data.getNextInputs(inputVals))break;
		if (!data.getTargetOutputs(targetVals))break;

		HANDLE_ERROR(cudaMemcpy(d_input, inputVals.data(), sizeof(float)* topology[0], cudaMemcpyHostToDevice));

		float* currInput = d_input;
		for (int i = 0;i < net.layers.size();i++) {
			net.layers[i].feedForward(currInput);
			currInput = net.layers[i].d_output;
		}

		HANDLE_ERROR(cudaMemcpy(resultVals.data(), finalLayer.d_output,sizeof(float)* resultVals.size(), cudaMemcpyDeviceToHost));

		showVectorVals("Inputs:", inputVals);
		showVectorVals("Outputs:", resultVals);
		showVectorVals("Targets:", targetVals);
		cout << endl;

	}

	for (Layer& layer : net.layers) {
		layer.freeDeviceMem();
	}

	cudaFree(d_input);
	cout << "Done" << endl;
	return 0;
}