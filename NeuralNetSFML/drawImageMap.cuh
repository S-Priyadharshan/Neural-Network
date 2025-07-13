#pragma once
#include <SFML//Graphics.hpp>
#include <vector>
#include "Fnn.cuh";
#include "SFML_includes.cuh"
#include"TrainingData.h"

using namespace std;

void drawImageMap(const std::vector<sf::Vector2f>& inputs,
    const std::vector<sf::Color>& colors,
    sf::RenderWindow& window,
    Net& net, float* d_input);