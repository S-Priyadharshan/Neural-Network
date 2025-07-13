#include "SFML_includes.cuh"
#include <vector>
#include <iomanip>
#include <sstream>
using namespace std;

void drawNeuralNet(sf::RenderWindow& window, sf::Font& font, vector<vector<sf::Vector2f>>& positions, vector<unsigned>& topology, vector<vector<float>>& activations,vector<float>&inputVals) {
	
	int CIRCLE_RADIUS = 32.f;

	window.clear(sf::Color::Black);

    for (size_t i = 0; i < positions.size() - 1; i++) {
        for (const auto& from : positions[i]) {
            for (const auto& to : positions[i + 1]) {
                sf::Vertex line[] = {
                    sf::Vertex(from, sf::Color::White),
                    sf::Vertex(to, sf::Color::White)
                };
                window.draw(line, 2, sf::Lines);
            }
        }
    }

    for (size_t i = 0; i < positions.size(); i++) {
        for (size_t j = 0; j < positions[i].size(); j++) {
            const auto& pos = positions[i][j];

            sf::CircleShape neuron(32.f);
            neuron.setOrigin(32.f, 32.f);
            neuron.setPosition(pos);
            neuron.setFillColor(sf::Color::White);
            window.draw(neuron);

            sf::Text text;
            text.setFont(font);
            text.setCharacterSize(14);
            text.setFillColor(sf::Color::Black);
            std::ostringstream ss;
            if (i == 0 && j<inputVals.size())
                ss << std::fixed << std::setprecision(2) <<inputVals[j];
            else
                ss << std::fixed << std::setprecision(2) << activations[i - 1][j];

            text.setString(ss.str());
            sf::FloatRect bounds = text.getLocalBounds();
            text.setOrigin(bounds.width / 2, bounds.height / 2 + bounds.top);
            text.setPosition(pos);
            window.draw(text);
        }
    }
}
