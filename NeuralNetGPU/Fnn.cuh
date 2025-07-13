#pragma once

#include <vector>
using namespace std;

__global__ void feedForwardKernel(const float* inputs, const float* weights, const float* bias, float* outputs, int numInputs);
__global__ void computeDelta(const float* predicted, const float* target, float* errors, int numOutputNeurons);
__global__ void computeOutputGradients(const float* outputVals, const float* errors, float* gradients, int numNeurons);
__global__ void computeHiddenLayerGradients(const float* hiddenOutputs, const float* nextWeights, const float* nextGradients, float* hiddenGradients, int numHiddenNeurons, int numNextNeurons);
__global__ void updateWeights(float* weights, const float* inputs, const float* gradients, float* prevDeltaWeights, int numInputs, int numNeurons, float eta, float alpha);
__global__ void computeRMSError(const float* deltas, float* rms, int size);
__global__ void updateBias(float* bias, const float* gradients, int numNeurons, float eta);

struct Layer {
    int numNeurons;
    int numInputs;
    std::vector<float> weights, bias, output, delta, gradients, prevDeltaWeights;

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

//struct Net {
//    vector<Layer> layers;
//    void Initialize(vector<unsigned>& topology);
//    vector<float> feedForward(vector<float>& inputVals);
//    vector<float> backProp(vector<float>& targetVals);
//};

struct Net {
    std::vector<Layer> layers;
    float eta = 0.1f;
    float alpha = 0.5f;

    float* d_input = nullptr;
    float* d_targetVals = nullptr;
    float* d_rms = nullptr;

    void Initialize(std::vector<unsigned>& topology);
    void allocateBuffers(int inputSize, int outputSize);
    void freeBuffers();

    void feedForward(float* d_input);
    void backPropagate(float* d_input, float* d_targetVals);
    void train(TrainingData& data, int epochs);
};
