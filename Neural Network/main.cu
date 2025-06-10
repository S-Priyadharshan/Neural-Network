#include<iostream>
#include<stdlib.h>
#include<ctime>
#include<vector>

using namespace std;

double randDouble() {
    return static_cast<double>(rand()) / RAND_MAX;
}

class Neuron {
private:
    double output;
    double bias;
    vector<double>weights;
public:
    // Constructor for neuron
    Neuron(int numInputs) {
        bias = randDouble();
        for (int i = 0;i < numInputs;i++) {
            weights.push_back(randDouble());
        }
    }

    void feedForward(const vector<double>& inputs) {
        output = bias;
        for (int i = 0;i < inputs.size();i++) {
            output += weights[i] * inputs[i];
        }
    }

    double getOutput() const{
        return output;
    }

    void printWeightAndBias()const {
        cout << "Weights: ";
        for (double w : weights)cout << w << " ";
        cout << "Bias: " << bias << "\n";
    }
};


int main() {
	srand(static_cast<unsigned>(time(0)));

    vector<double>inputs = { randDouble(),randDouble() ,randDouble() };

    vector<Neuron>layers;
    for (int i = 0;i < 3;i++) {
        layers.emplace_back(Neuron(inputs.size()));
    }

    for (int i = 0;i < layers.size();i++) {
        layers[i].feedForward(inputs);
        cout << "Neuron: " << i + 1 << " Output: " << layers[i].getOutput() << "\n";
        layers[i].printWeightAndBias();
    }

    return 0;
}

