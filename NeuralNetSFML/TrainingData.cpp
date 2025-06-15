#include"TrainingData.h"
#include<sstream>
#include<fstream>
#include<iostream>

using namespace std;

TrainingData::TrainingData(const string& filename) {
	file.open(filename);
	if (!file.is_open()) {
		cerr << "Error opening file\n";
		exit(1);
	}
}

bool TrainingData::isEof() const {
	return file.eof();
}

void TrainingData::getTopology(vector<unsigned>& topology) {
	topology.clear();
	string line, label;
	getline(file, line);
	istringstream ss(line);
	ss >> label;
	if (label != "topology:") {
		cerr << "Expected topology line\n";
		exit(1);
	}
	unsigned n;
	while (ss >> n)topology.push_back(n);
}

bool TrainingData::getNextInputs(vector<float>& inputVals) {
	inputVals.clear();
	string line, label;
	getline(file, line);
	istringstream ss(line);
	ss >> label;
	if (label != "in:") return false;
	float val;
	while (ss >> val) inputVals.push_back(val);
	return true;
}

bool TrainingData::getTargetOutputs(vector<float>& targetVals) {
	targetVals.clear();
	string line, label;
	getline(file, line);
	istringstream ss(line);
	ss >> label;
	if (label != "out:")return false;
	float val;
	while (ss >> val)targetVals.push_back(val);
	return true;
}

void TrainingData::reset() {
	file.clear();
	file.seekg(0, ios::beg);
}