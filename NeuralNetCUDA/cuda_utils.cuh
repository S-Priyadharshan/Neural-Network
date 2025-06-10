#pragma once
#include <iostream>
#include <cuda_runtime.h>

#define HANDLE_ERROR(err) (HandleError(err,__FILE__,__LINE__))

inline void HandleError(cudaError_t err, const char* file, int line) {
	if (err != cudaSuccess) {
		std::cerr << "[CUDA ERROR]" << cudaGetErrorString(err)
			<< " at " << file << ":" << line << std::endl;
		exit(EXIT_FAILURE);
	}
}


