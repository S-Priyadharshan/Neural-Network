#include<iostream>
#include<vector>
#include<cmath>
#include<cstdlib>
#include<cassert>
#include<fstream>
#include<sstream>
#include<chrono>
using namespace std;
using namespace std::chrono;


class TrainingData {
public:
	TrainingData(const string filename);
	bool isEof(void) { return trainingDataFile.eof(); }
	void getTopology(vector<unsigned>& topology);
	unsigned getNextInputs(vector<double>& inputVals);
	unsigned getTargetOutputs(vector<double>& targetOutputVals);

	void reset() {
		trainingDataFile.clear();          // clear EOF and fail bits
		trainingDataFile.seekg(0, ios::beg); // seek to beginning
	}

private:
	ifstream trainingDataFile;
};

TrainingData::TrainingData(const string filename) {
	trainingDataFile.open(filename.c_str());
}

void TrainingData::getTopology(vector<unsigned>& topology) {
	string line;
	getline(trainingDataFile, line);
	string label;
	istringstream iss(line);
	iss >> label;
	if (label != "topology:") exit(1);
	unsigned n;
	while (iss >> n)
		topology.push_back(n);
}

unsigned TrainingData::getNextInputs(vector<double>& inputVals) {
	inputVals.clear();
	string line;
	getline(trainingDataFile, line);
	string label;
	istringstream iss(line);
	iss >> label;
	if (label == "in:") {
		double oneValue;
		while (iss >> oneValue)
			inputVals.push_back(oneValue);
	}
	return inputVals.size();
}

unsigned TrainingData::getTargetOutputs(vector<double>& targetOutputVals) {
	targetOutputVals.clear();
	string line;
	getline(trainingDataFile, line);
	string label;
	istringstream iss(line);
	iss >> label;
	if (label == "out:") {
		double oneValue;
		while (iss >> oneValue)
			targetOutputVals.push_back(oneValue);
	}
	return targetOutputVals.size();
}


struct Connection {
	double weight;
	double delWeights;
};

class Neuron;

typedef vector<Neuron> Layer;

class Neuron {
public:
	Neuron(unsigned numOutputs, unsigned Index); // here num outputs is like the amount of neurons in the next layer
	void setOutputVal(double val) { outputVal = val; }
	double getOutputVal() const { return outputVal; }
	void feedForward(const Layer& prevLayer);
	void calcOutputGradients(double targetVal);
	void calcHiddenGradient(const Layer& nextLayer);
	void updateInputWeights(Layer& prevLayer);

private:
	double outputVal;
	unsigned n_Index;
	static double randWeight(void) { return static_cast<double>(rand()) / RAND_MAX; }
	vector<Connection> outputWeights;
	double sumDow(const Layer& nextLayer);
	static double transferFunction(double x);
	static double transferFunctionDerivative(double x);
	double gradient;
	static double eta;
	static double alpha;
};

double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;

Neuron::Neuron(unsigned numOutputs, unsigned Index) {
	
	for (int c = 0;c < numOutputs;c++) {
		outputWeights.push_back(Connection());
		outputWeights.back().weight = randWeight();
	}

	n_Index = Index;
}

//double Neuron::transferFunction(double x) {
//	return x > 0.0 ? x : 0.0;
//}
//
//double Neuron::transferFunctionDerivative(double x) {
//	return x > 0.0 ? 1.0 : 0.0;
//}

double Neuron::transferFunction(double x) {
	return 1.0 / (1.0 + exp(-x)); // Sigmoid
}

double Neuron::transferFunctionDerivative(double x) {
	// derivative in terms of outputVal (x here should be outputVal)
	return x * (1.0 - x);
}

void Neuron::updateInputWeights(Layer& prevLayer) {
	for (unsigned i = 0;i < prevLayer.size();i++) {
		Neuron& neuron = prevLayer[i];
		double oldDeltaWeight = neuron.outputWeights[n_Index].delWeights;
		double newDeltaWeight = eta * neuron.getOutputVal() * gradient + alpha * oldDeltaWeight;
		neuron.outputWeights[n_Index].delWeights = newDeltaWeight;
		neuron.outputWeights[n_Index].weight += newDeltaWeight;
	}
}

double Neuron::sumDow(const Layer& nextLayer) {
	double sum = 0.0;

	for (unsigned n = 0;n < nextLayer.size() - 1;n++) {
		sum += outputWeights[n].weight * nextLayer[n].gradient;
	}
	
	return sum;
}


void Neuron::feedForward(const Layer& prevLayer) {
	double sum = 0.0;

	for (unsigned i = 0;i < prevLayer.size();i++) {
		sum += prevLayer[i].getOutputVal() * prevLayer[i].outputWeights[n_Index].weight;
	}

	outputVal = Neuron::transferFunction(sum); //since its static
}

void Neuron::calcOutputGradients(double targetVal) {
	double delta = targetVal - outputVal;
	gradient = delta * transferFunctionDerivative(outputVal);
}

void Neuron::calcHiddenGradient(const Layer& nextLayer) {
	double gow = sumDow(nextLayer);
	gradient = gow * transferFunctionDerivative(outputVal);
}
//````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````

class Net {
public:
	Net(const vector<unsigned>& topology);//constructor
	void feedForward(const vector<double>& inputVals);
	void backProp(const vector<double>& targetVals);
	void getResults(vector<double>& resultVals)const;
	double getRecentAverageError(void)const { return avgError; }

private:
	vector<Layer> layers; // this is vector<vector<Neuron>> layers
	double error;         // basically layout of the whole network
	double avgError;
	static double avgErrorSmoothingFactor;
};

double Net::avgErrorSmoothingFactor = 100.0;

Net::Net(const vector<unsigned>& topology) {
	
	unsigned numLayers = topology.size();
	for (int i = 0;i < numLayers;i++) {	
		layers.push_back(Layer());
		unsigned numOutputs = i == topology.size() - 1 ? 0 : topology[i + 1]; // basically telling how many neuron
																			  // in the next layer
		for (int n = 0;n <= topology[i];n++) {
			layers.back().push_back(Neuron(numOutputs,n));
			cout << "Made a Neuron!" << endl;
		}

		layers.back().back().setOutputVal(1.0);
	}
}

void Net::feedForward(const vector<double>& inputVals) {
	assert(inputVals.size() == layers[0].size() - 1); 

	for (int i = 0;i < inputVals.size();i++) {
		layers[0][i].setOutputVal(inputVals[i]); // gotta use setOutputVal since outputval is private to neuron
	}

	for (unsigned i = 1;i < layers.size();i++) {
		Layer& prevLayer = layers[i - 1]; // setting a "reference" to prev layer
		/*for (unsigned n = 0;n < layers[i].size();n++) {
			layers[i][n].feedForward(prevLayer);
		}*/
		for (unsigned n = 0;n < layers[i].size()-1;n++) {
			layers[i][n].feedForward(prevLayer);
		}
	}

}

void Net::backProp(const vector<double>& targetVals) {
	// net error (root mean square)
	Layer& outputLayer = layers.back();
	double error = 0.0;

	for (int i = 0;i < targetVals.size();i++) {
		double delta = targetVals[i] - outputLayer[i].getOutputVal();
		error += delta * delta;
	}

	error /= outputLayer.size() - 1;
	error = sqrt(error);

	avgError = (avgError * avgErrorSmoothingFactor + error) / (avgErrorSmoothingFactor + 1.0);

	//gradients

	for (int i = 0;i < outputLayer.size() - 1;i++) {
		outputLayer[i].calcOutputGradients(targetVals[i]);
	}

	for (int n = layers.size() - 2;n > 0;n--) {
		Layer& hiddenLayer = layers[n];
		Layer& nextLayer = layers[n + 1];

		/*for (int i = 0;i < hiddenLayer.size();i++) {
			hiddenLayer[i].calcHiddenGradient(nextLayer);
		}*/
		for (int i = 0;i < hiddenLayer.size()-1;i++) {
			hiddenLayer[i].calcHiddenGradient(nextLayer);
		}
	}

	for (int i = layers.size() - 1;i>0;i--) {
		Layer& layer = layers[i];
		Layer& prevLayer = layers[i - 1];

		for (int j = 0;j < layer.size()-1;j++) {
			layer[j].updateInputWeights(prevLayer);
		}
	}

}


void Net::getResults(vector<double>& resultVals)const {
	resultVals.clear();

	for (int i = 0;i < layers.back().size() - 1;i++) {
		resultVals.push_back(layers.back()[i].getOutputVal());
	}
}

void showVectorVals(string label, vector<double>& v)
{
	cout << label << " ";
	for (unsigned i = 0; i < v.size(); ++i) {
		cout << v[i] << " ";
	}

	cout << endl;
}

int main() {
	TrainingData trainData("trainingData.txt");  // XOR file

	// e.g., { 2, 2, 1 }
	vector<unsigned> topology;
	trainData.getTopology(topology);

	Net myNet(topology);

	vector<double> inputVals, targetVals, resultVals;
	int trainingPass = 0;
	const int maxEpochs = 5000;  // Set how many times you want to train over the dataset

	auto start = high_resolution_clock::now();

	for (int epoch = 0; epoch < maxEpochs; ++epoch) {
		// Reset file to beginning except for topology line
		trainData.reset();
		trainData.getTopology(topology);  // skip topology line again

		while (!trainData.isEof()) {
			++trainingPass;

			if (trainData.getNextInputs(inputVals) != topology[0]) {
				break;
			}

			myNet.feedForward(inputVals);

			myNet.getResults(resultVals);

			trainData.getTargetOutputs(targetVals);
			if (targetVals.size() != topology.back()) {
				break;
			}

			myNet.backProp(targetVals);
		}

		if (epoch % 100 == 0) { // print progress every 100 epochs
			cout << "Epoch " << epoch << " recent average error: "
				<< myNet.getRecentAverageError() << endl;
		}
	}

	auto end = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(end - start);


	cout << "Training complete after " << maxEpochs << " epochs." << endl;

	cout << "Training Time: " << duration.count() << " ms" << endl;

	// Optionally test final outputs
	trainData.reset();
	trainData.getTopology(topology);

	cout << "\nFinal outputs:\n";

	while (!trainData.isEof()) {
		if (trainData.getNextInputs(inputVals) != topology[0]) {
			break;
		}

		myNet.feedForward(inputVals);
		myNet.getResults(resultVals);
		trainData.getTargetOutputs(targetVals);

		showVectorVals("Inputs:", inputVals);
		showVectorVals("Outputs:", resultVals);
		showVectorVals("Targets:", targetVals);
		cout << endl;
	}

	cout << "Done" << endl;
}

