#include "SFML_includes.cuh"
#include <vector>
#include <iomanip>
#include <sstream>
using namespace std;

void drawNeuralNet(sf::RenderWindow& window, sf::Font& font, vector<vector<sf::Vector2f>>& positions, vector<unsigned>& topology, vector<vector<float>>& activations) {
	
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
            neuron.setFillColor(sf::Color::Green);
            window.draw(neuron);

            sf::Text text;
            text.setFont(font);
            text.setCharacterSize(14);
            text.setFillColor(sf::Color::Black);
            std::ostringstream ss;
            if (i == 0)
                ss << "IN";
            else
                ss << std::fixed << std::setprecision(2) << activations[i - 1][j];
            text.setString(ss.str());
            sf::FloatRect bounds = text.getLocalBounds();
            text.setOrigin(bounds.width / 2, bounds.height / 2 + bounds.top);
            text.setPosition(pos);
            window.draw(text);
        }
    }

    window.display();
}

//void drawNeuralNet(std::vector<unsigned>&topology,vector<vector<float>>&weights, vector<vector<float>>& biases,vector<vector<float>>&activations) {
//	int SCREEN_WIDTH = 800;
//	int SCREEN_HEIGHT = 600;
//	/*sf::RenderWindow window(sf::VideoMode(SCREEN_WIDTH, SCREEN_HEIGHT), "Neural Network");
//	window.setFramerateLimit(60);*/
//
//	int CIRCLE_RADIUS = 32.f;
//
//	/*std::vector<std::vector<sf::Vector2f>> positions(topology.size());
//
//	for (unsigned int i = 0;i < topology.size();i++) {
//		float x = (SCREEN_WIDTH / (topology.size() + 1)) * (i + 1);
//		for (unsigned int j = 0;j < topology[i];j++) {
//			float y = (SCREEN_HEIGHT / (topology[i] + 1)) * (j + 1);
//			positions[i].emplace_back(x, y);
//		}
//	}*/
//
//	sf::Font font;
//	if (!font.loadFromFile("Monospace.ttf")) {
//		cout << "Unable to load font\n";
//		return;
//	}
//
//	while (window.isOpen()) {
//		sf::Event e;
//		
//		while (window.pollEvent(e)) {
//			if (e.type == sf::Event::Closed) {
//				window.close();
//			}
//
//			if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape)) {
//				window.close();
//			}
//		}
//
//		window.clear(sf::Color::Black);
//
//		for (size_t i = 0;i < positions.size() - 1;i++) {
//			for (const auto& from : positions[i]) {
//				for (const auto& to : positions[i + 1]) {
//					sf::Vertex line[] = {
//						sf::Vertex(from,sf::Color::White),
//						sf::Vertex(to, sf::Color::White)
//					};
//					window.draw(line, 2, sf::Lines);
//				}
//			}
//		}
//			
//		for (size_t i = 0; i < positions.size(); i++) {
//			for (size_t j = 0; j < positions[i].size(); j++) {
//				const auto& pos = positions[i][j];
//
//				sf::CircleShape neuron(CIRCLE_RADIUS);
//				neuron.setOrigin(CIRCLE_RADIUS, CIRCLE_RADIUS);
//				neuron.setPosition(pos);
//				neuron.setFillColor(sf::Color::Green);
//				window.draw(neuron);
//
//				sf::Text text;
//				text.setFont(font);
//				text.setCharacterSize(14);
//				text.setFillColor(sf::Color::Black);
//
//				std::ostringstream ss;
//
//				if (i == 0) {
//					ss << "IN";
//				}
//				else {
//					ss << fixed << setprecision(2) << activations[i - 1][j];
//				}
//				text.setString(ss.str());
//
//				sf::FloatRect bounds = text.getLocalBounds();
//				text.setOrigin(bounds.width / 2, bounds.height / 2 + bounds.top);
//				text.setPosition(pos);
//
//				window.draw(text);
//			}
//		}
//
//		window.display();
//	}
//}