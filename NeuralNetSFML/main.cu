#include<iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "device_atomic_functions.h"
#include <random>
#include <chrono>

#include"SFML_includes.cuh"
#include"TrainingData.h"
#include "Fnn.cuh";
#include "drawImageMap.cuh";

using namespace std;

template<typename T>

T clamp(T val, T low, T high) {
	return std::max(low, std::min(val, high));
}

const std::chrono::microseconds FRAME_DURATION(100000);

int main() {
    TrainingData data("trainingData.txt");
    vector<unsigned> topology;
    data.getTopology(topology);

    Net net;
    net.Initialize(topology);
    net.allocateBuffers(topology[0], topology.back());
    for (Layer& layer : net.layers) {
        layer.allocateOnDevice();
    }

    vector<float> inputVals, outputVals;
    vector<sf::Vector2f> inputs;
    vector<sf::Color> colors;
    bool train = false;

    float* d_input = nullptr;
    float* d_targetVals = nullptr;
    HANDLE_ERROR(cudaMalloc(&d_input, sizeof(float) * 2));
    HANDLE_ERROR(cudaMalloc(&d_targetVals, sizeof(float) * 3));

    sf::RenderWindow window(sf::VideoMode(2 * 800, 600), "Neural Net");
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
        float x = (SCREEN_WIDTH / (topology.size() + 1)) * (i + 1);
        for (unsigned int j = 0; j < topology[i]; j++) {
            float y = (SCREEN_HEIGHT / (topology[i] + 1)) * (j + 1);
            positions[i].emplace_back(x, y);
        }
    }

    std::chrono::steady_clock::time_point previous_time = std::chrono::steady_clock::now();
    std::chrono::microseconds lag(0);

    while (window.isOpen()) {
        auto current_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(current_time - previous_time);
        previous_time = current_time;
        lag += elapsed;

        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
            else if (event.type == sf::Event::KeyReleased && event.key.code == sf::Keyboard::Enter) {
                train = true;
            }
            else if (event.type == sf::Event::MouseButtonPressed && event.mouseButton.button == sf::Mouse::Left) {
                int mx = event.mouseButton.x, my = event.mouseButton.y;
                if (mx >= SCREEN_WIDTH && mx < 2 * SCREEN_WIDTH && my >= 0 && my < SCREEN_HEIGHT) {
                    float dot_x = (mx - SCREEN_WIDTH) / static_cast<float>(SCREEN_WIDTH);
                    float dot_y = my / static_cast<float>(SCREEN_HEIGHT);
                    sf::Color color;
                    vector<float> label;
                    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Q)) {
                        color = sf::Color::Red; label = { 1, 0, 0 };
                    }
                    else if (sf::Keyboard::isKeyPressed(sf::Keyboard::W)) {
                        color = sf::Color::Green; label = { 0, 1, 0 };
                    }
                    else if (sf::Keyboard::isKeyPressed(sf::Keyboard::E)) {
                        color = sf::Color::Blue; label = { 0, 0, 1 };
                    }
                    else break;
                    inputs.emplace_back(dot_x, dot_y);
                    colors.emplace_back(color);
                }
            }
        }

        while (lag >= FRAME_DURATION) {
            lag -= FRAME_DURATION;
            if (train && !inputs.empty() && inputs.size() == colors.size()){
                for (int i = 0; i < 50; ++i) {
                    int idx = rand() % inputs.size();
                    inputVals = { inputs[idx].x, inputs[idx].y };
                    sf::Color c = colors[idx];
                    outputVals = {
                        c == sf::Color::Red ? 1.0f : 0.0f,
                        c == sf::Color::Green ? 1.0f : 0.0f,
                        c == sf::Color::Blue ? 1.0f : 0.0f
                    };
                    HANDLE_ERROR(cudaMemcpy(d_input, inputVals.data(), sizeof(float) * 2, cudaMemcpyHostToDevice));
                    HANDLE_ERROR(cudaMemcpy(d_targetVals, outputVals.data(), sizeof(float) * 3, cudaMemcpyHostToDevice));
                    net.feedForward(d_input);
                    net.backPropagate(d_input, d_targetVals);
                }

                for (auto& layer : net.layers) {
                    HANDLE_ERROR(cudaMemcpy(layer.output.data(), layer.d_output, sizeof(float) * layer.output.size(), cudaMemcpyDeviceToHost));
                }
            }
        }

        window.clear();
        if (inputVals.size() != topology[0]) {
            inputVals.assign(topology[0], 0.0f);
        }
        drawNeuralNet(window, font, positions, topology, net.getAllActivations(), inputVals);
    
        drawImageMap(inputs, colors, window, net,d_input);

        window.display();
    }

    for (Layer& layer : net.layers) layer.freeDeviceMem();
    cudaFree(d_input);
    cudaFree(d_targetVals);
    return 0;
}





