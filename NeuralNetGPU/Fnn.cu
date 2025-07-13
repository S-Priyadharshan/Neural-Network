#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "device_atomic_functions.h"
#include "TrainingData.h"
#include <random>
#include <cassert>

#include"cuda_utils.cuh";
#include "Fnn.cuh";

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

void Net::Initialize(vector<unsigned>& topology) {
	layers.clear();

	for (int i = 0;i < topology.size() - 1;i++) {
		Layer layer;
		layer.Initialize(topology[i], topology[i + 1]);
		layers.push_back(layer);
	}
}

void Net::allocateBuffers(int inputSize, int outputSize) {
	HANDLE_ERROR(cudaMalloc(&d_input, sizeof(float) * inputSize));
	HANDLE_ERROR(cudaMalloc(&d_targetVals, sizeof(float) * outputSize));
	HANDLE_ERROR(cudaMalloc(&d_rms, sizeof(float)));
	HANDLE_ERROR(cudaMemset(d_rms, 0, sizeof(float)));
}

void Net::freeBuffers() {
	cudaFree(d_input);
	cudaFree(d_targetVals);
	cudaFree(d_rms);
}

void Net::feedForward(float* d_input) {
	float* currInput = d_input;
	for (Layer& layer : layers) {
		layer.feedForward(currInput);
		currInput = layer.d_output;
	}
}

void Net::backPropagate(float* d_input, float* d_targetVals) {
	Layer& outputLayer = layers.back();
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

	computeOutputGradients << <1, numOutputNeurons >> > (
		outputLayer.d_output,
		outputLayer.d_delta,
		outputLayer.d_gradients,
		numOutputNeurons
		);

	for (int i = layers.size() - 2; i >= 0; i--) {
		Layer& hiddenLayer = layers[i];
		Layer& nextLayer = layers[i + 1];
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

	for (int i = layers.size() - 1; i >= 0; i--) {
		Layer& currLayer = layers[i];
		const float* prevOutputs = (i == 0) ? d_input : layers[i - 1].d_output;
		int prevSize = (i == 0) ? layers[0].numInputs : layers[i - 1].numNeurons;

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

void Net::train(TrainingData& data, int epochs) {
	std::vector<float> inputVals, targetVals;
	vector<unsigned>topology;
	for (int epoch = 0; epoch < epochs; ++epoch) {
		data.reset();
		data.getTopology(topology);
		while (!data.isEof()) {
			if (!data.getNextInputs(inputVals)) break;
			if (!data.getTargetOutputs(targetVals)) break;
			if (inputVals.size() != topology[0]) break;

			HANDLE_ERROR(cudaMemcpy(d_input, inputVals.data(), sizeof(float) * inputVals.size(), cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMemcpy(d_targetVals, targetVals.data(), sizeof(float) * targetVals.size(), cudaMemcpyHostToDevice));

			feedForward(d_input);
			backPropagate(d_input, d_targetVals);
		}
	}
}