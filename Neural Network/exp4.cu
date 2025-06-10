#include <cuda_runtime.h>
#include <stdio.h>
#include<stdlib.h>
#include<iostream>
#include <chrono>  // Add this to the top for CPU timing

// Macro for checking CUDA errors following a CUDA API call
#define CHECK(call)                                                          \
{                                                                           \
    const cudaError_t error = call;                                         \
    if (error != cudaSuccess)                                               \
    {                                                                       \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                       \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1);                                                            \
    }                                                                       \
}



/*
 * This example demonstrates the use of CUDA managed memory to implement matrix
 * addition. In this example, arbitrary pointers can be dereferenced on the host
 * and device. CUDA will automatically manage the transfer of data to and from
 * the GPU as needed by the application. There is no need for the programmer to
 * use cudaMemcpy, cudaHostGetDevicePointer, or any other CUDA API involved with
 * explicitly transferring data. In addition, because CUDA managed memory is not
 * forced to reside in a single place it can be transferred to the optimal
 * memory space and not require round-trips over the PCIe bus every time a
 * cross-device reference is performed (as is required with zero copy and UVA).
 */

void initialData(float* ip, const int size)
{
    int i;

    for (i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }

    return;
}

void sumMatrixOnHost(float* A, float* B, float* C, const int nx, const int ny)
{
    float* ia = A;
    float* ib = B;
    float* ic = C;

    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix = 0; ix < nx; ix++)
        {
            ic[ix] = ia[ix] + ib[ix];
        }

        ia += nx;
        ib += nx;
        ic += nx;
    }

    return;
}

void checkResult(float* hostRef, float* gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("host %f gpu %f\n", hostRef[i], gpuRef[i]);
            break;
        }
    }

    if (!match)
    {
        printf("Arrays do not match.\n\n");
    }
}


__global__ void sumMatrixGPU(float* MatA, float* MatB, float* MatC, int nx, int ny)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;

    if (ix < nx && iy < ny)
    {
        int idx = iy * nx + ix;
        MatC[idx] = MatA[idx] + MatB[idx];
    }
}

int main(int argc, char** argv)
{
    printf("%s Starting ", argv[0]);

    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    int nx, ny;
    int ishift = 12;

    if (argc > 1) ishift = atoi(argv[1]);

    nx = ny = 1 << ishift;

    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    float* A, * B, * hostRef, * gpuRef;
    CHECK(cudaMallocManaged((void**)&A, nBytes));
    CHECK(cudaMallocManaged((void**)&B, nBytes));
    CHECK(cudaMallocManaged((void**)&gpuRef, nBytes));
    CHECK(cudaMallocManaged((void**)&hostRef, nBytes));

    // Initialize data (no timing)
    initialData(A, nxy);
    initialData(B, nxy);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    auto cpu_start = std::chrono::high_resolution_clock::now();
    sumMatrixOnHost(A, B, hostRef, nx, ny);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = cpu_end - cpu_start;
    printf("sumMatrix on host:\t %f sec\n", cpu_duration.count());

    // sumMatrix on host (CPU) no timing
    sumMatrixOnHost(A, B, hostRef, nx, ny);

    // Setup CUDA events for GPU timing
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // Warm-up kernel
    sumMatrixGPU << <grid, block >> > (A, B, gpuRef, 1, 1);

    // GPU timing with CUDA events
    CHECK(cudaEventRecord(start));
    sumMatrixGPU << <grid, block >> > (A, B, gpuRef, nx, ny);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("sumMatrix on gpu :\t %f sec <<<(%d,%d), (%d,%d)>>> \n",
        milliseconds / 1000.0f, grid.x, grid.y, block.x, block.y);

    printf("Sample output from GPU result:\n");
    for (int i = 0; i < 10; ++i) {
        printf("gpuRef[%d] = %f\n", i, gpuRef[i]);
    }

    // check kernel error
    CHECK(cudaGetLastError());

    // check device results
    checkResult(hostRef, gpuRef, nxy);

    // free memory
    CHECK(cudaFree(A));
    CHECK(cudaFree(B));
    CHECK(cudaFree(hostRef));
    CHECK(cudaFree(gpuRef));

    // destroy CUDA events
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    // reset device
    CHECK(cudaDeviceReset());

    return 0;
}
