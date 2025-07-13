#include "drawImageMap.cuh";
#include "SFML_includes.cuh";
#include <iostream>
#include "Fnn.cuh";

template<typename T>

T clamp(T val, T low, T high) {
    return std::max(low, std::min(val, high));
}

void drawImageMap(
    const std::vector<sf::Vector2f>& inputs,
    const std::vector<sf::Color>& colors,
    sf::RenderWindow& window,
    Net& net,
    float* d_input
) {
    const int SCREEN_WIDTH = 800;
    const int SCREEN_HEIGHT = 600;
    const int OUTPUTS_WIDTH = 64;
    const int OUTPUTS_HEIGHT = 64;

    static sf::Image output_img;
    static sf::Texture output_texture;
    static sf::Sprite output_sprite;
    static bool initialized = false;

    if (!initialized) {
        output_img.create(OUTPUTS_WIDTH, OUTPUTS_HEIGHT, sf::Color::Black);
        output_texture.loadFromImage(output_img);
        output_sprite.setTexture(output_texture);
        output_sprite.setPosition(SCREEN_WIDTH, 0);
        output_sprite.setScale(
            static_cast<float>(SCREEN_WIDTH) / OUTPUTS_WIDTH,
            static_cast<float>(SCREEN_HEIGHT) / OUTPUTS_HEIGHT
        );
        initialized = true;
    }

    for (int x = 0; x < OUTPUTS_WIDTH; ++x) {
        for (int y = 0; y < OUTPUTS_HEIGHT; ++y) {
            float input_x = static_cast<float>(x) / OUTPUTS_WIDTH;
            float input_y = static_cast<float>(y) / OUTPUTS_HEIGHT;

            std::vector<float> input = { input_x, input_y };
            HANDLE_ERROR(cudaMemcpy(d_input, input.data(), sizeof(float) * 2, cudaMemcpyHostToDevice));

            net.feedForward(d_input);

            const Layer& outputLayer = net.layers.back();
            std::vector<float> host_output(outputLayer.output.size());
            HANDLE_ERROR(cudaMemcpy(
                host_output.data(),
                outputLayer.d_output,
                sizeof(float) * host_output.size(),
                cudaMemcpyDeviceToHost
            ));

            int r = static_cast<int>(255 * clamp(host_output[0], 0.0f, 1.0f));
            int g = static_cast<int>(255 * clamp(host_output[1], 0.0f, 1.0f));
            int b = static_cast<int>(255 * clamp(host_output[2], 0.0f, 1.0f));

            output_img.setPixel(x, y, sf::Color(r, g, b));
        }
    }

    
    output_texture.update(output_img);
    window.draw(output_sprite);

    
    sf::CircleShape dot_shape(8.0f);  
    dot_shape.setOrigin(dot_shape.getRadius(), dot_shape.getRadius());
    dot_shape.setOutlineColor(sf::Color::Black);
    dot_shape.setOutlineThickness(-1.0f);

    for (size_t i = 0; i < inputs.size(); ++i) {
        sf::Vector2f pos = inputs[i];
        dot_shape.setFillColor(colors[i]);
        dot_shape.setPosition(SCREEN_WIDTH + pos.x * SCREEN_WIDTH, pos.y * SCREEN_HEIGHT);
        window.draw(dot_shape);
    }
}

