#pragma once

#include <SFML/Graphics.hpp>
#include <iostream>
#include <vector>
#include<cuda_runtime.h>

using namespace std;

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

inline void HandleError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << cudaGetErrorString(err) << " in " << file << " at line " << line << std::endl;
        exit(EXIT_FAILURE);
    }
}

void drawNeuralNet(sf::RenderWindow& window, sf::Font& font, vector<vector<sf::Vector2f>>&positions, vector<unsigned>& topology, vector<vector<float>>& activations,vector<float>&inputVals);
