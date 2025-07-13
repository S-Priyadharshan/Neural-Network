#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "device_atomic_functions.h"
#include "TrainingData.h"
#include <random>
#include <cassert>

#include "cuda_utils.cuh";
#include "Fnn.cuh";

using namespace std;

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
 
    std::vector<unsigned> topology;
    data.getTopology(topology);
    
    Net net;
    net.Initialize(topology);
    net.allocateBuffers(topology[0], topology.back());


    for (auto& layer : net.layers)
        layer.allocateOnDevice();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    net.train(data,5000);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    std::cout << "Training Time: " << time / 1000 << " s\n";

    std::vector<float> inputVals, targetVals, resultVals(topology.back());
    Layer& finalLayer = net.layers.back();

    data.reset();
    data.getTopology(topology);

    while (!data.isEof()) {
        if (!data.getNextInputs(inputVals)) break;
        if (!data.getTargetOutputs(targetVals)) break;

        HANDLE_ERROR(cudaMemcpy(net.d_input, inputVals.data(), sizeof(float) * topology[0], cudaMemcpyHostToDevice));
        net.feedForward(net.d_input);
        HANDLE_ERROR(cudaMemcpy(resultVals.data(), finalLayer.d_output, sizeof(float) * resultVals.size(), cudaMemcpyDeviceToHost));

        showVectorVals("Inputs:", inputVals);
        showVectorVals("Outputs:", resultVals);
        showVectorVals("Targets:", targetVals);
        std::cout << std::endl;
    }

    for (auto& layer : net.layers)
        layer.freeDeviceMem();
    net.freeBuffers();

    std::cout << "Done\n";
    return 0;
}

