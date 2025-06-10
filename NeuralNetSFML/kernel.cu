#include<iostream>

#include "SFML/Graphics.hpp"

int main() {
	sf::RenderWindow window(sf::VideoMode(1280, 720), "Test Run");
	window.setFramerateLimit(60);

	sf::RectangleShape rect;

	sf::Vector2f rectanglePosition(1280 / 2, 720 / 2);

	rect.setPosition(rectanglePosition);
	rect.setSize(sf::Vector2f(200, 200));

	float xv = 3;
	float yv = 3;

	while (window.isOpen()) {
		sf::Event e;
		while (window.pollEvent(e)) {
			if (e.type == sf::Event::Closed) window.close();

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape)) window.close();
		}

		if (rectanglePosition.x < 0 || rectanglePosition.x>1280 - 200)xv *= -1;
		if (rectanglePosition.y < 0 || rectanglePosition.y>720 - 200)yv *= -1;

		rectanglePosition.x += xv;
		rectanglePosition.y += yv;
		rect.setPosition(rectanglePosition);

		window.clear();
		window.draw(rect);
		window.display();
	}
}